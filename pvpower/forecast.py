import logging
from appdirs import site_data_dir
from threading import Thread
from datetime import datetime, timedelta
from typing import Optional
from pvpower.weather_forecast import WeatherStation, WeatherForecast
from pvpower.traindata import LabelledWeatherForecast, TrainSampleLog
from pvpower.estimator import Estimator
from pvpower.estimator import CoreVectorizer,PlusVisibilityVectorizer
from pvpower.estimator import PlusVisibilityCloudCoverVectorizer, PlusCloudCoverVectorizer, PlusSunshineVectorizer
from pvpower.estimator import PlusVisibilitySunshineVectorizer, PlusVisibilityFogCloudCoverVectorizer, FullVectorizer
from pvpower.tester import Tester


class Trainer:

    def train(self, train_log: TrainSampleLog, num_rounds: int = 10) -> Estimator:
        samples = train_log.all()

        vectorizer_map = {
            "core": CoreVectorizer(),
            "+visibility": PlusVisibilityVectorizer(),
            "+sunshine": PlusSunshineVectorizer(),
            "+cloudcover": PlusCloudCoverVectorizer(),
            "+visibility +sunshine": PlusVisibilitySunshineVectorizer(),
            "+visibility +cloudcover": PlusVisibilityCloudCoverVectorizer(),
            "+visibility +fog +cloudcover": PlusVisibilityFogCloudCoverVectorizer(),
            "+visibility +fog +cloudcover +sunshine": FullVectorizer(),
        }

        lowest_score = 10000
        best_estimator = Estimator(CoreVectorizer())

        days = len(set([sample.time_utc.strftime("%Y.%m.%d") for sample in samples]))
        report = "tested with " + str(len(samples)) + " cleaned samples (" + str(days) + " days; " + str(num_rounds) + " test rounds per variant)" + "\n"
        report += "VARIANT ................................. SCORE ....... DISTRIBUTION\n"
        for variant, vectorizer in vectorizer_map.items():
            estimator = Estimator(vectorizer)
            test_reports = Tester(samples).evaluate(estimator, rounds=num_rounds)
            median_report = test_reports[int(len(test_reports)*0.5)]
            score_str = str(round(median_report.score))
            distribution = str(round(test_reports[0].score)) + ", " + str(round(test_reports[1].score)) + ", " + str(round(test_reports[2].score)) + ", " + str(round(test_reports[3].score)) + ", ..., " + str(round(test_reports[int(len(test_reports)*0.5)].score)) + ", ..., " + str(round(test_reports[-4].score)) + ", " + str(round(test_reports[-3].score)) + ", " + str(round(test_reports[-2].score)) + ", " + str(round(test_reports[-1].score))
            report += variant + " " + "".join(["."] * (45 - (len(variant)+len(score_str)))) + " " + score_str + " ....... " + distribution + "\n"
            if median_report.score < lowest_score:
                best_estimator = estimator
                lowest_score = median_report.score
        logging.info(report)
        return best_estimator



class ValueRecorder:

    def __init__(self, window_size_min: int = 10):
        self.__start_time = datetime.now()
        self.__end_time = datetime.now() + timedelta(minutes=window_size_min)
        self.time = self.__start_time + timedelta(minutes=round(window_size_min/2))
        self.__power_values = []

    def empty(self) -> bool:
        return len(self.__power_values) == 0

    def is_expired(self):
        return datetime.now() > self.__end_time

    def add(self, value: int):
        self.__power_values.append(value)

    @property
    def average(self) -> Optional[int]:
        if len(self.__power_values) == 0:
            return None
        else:
            return int(sum(self.__power_values) / len(self.__power_values))

    def __str__(self):
        return self.__start_time.strftime("%Y.%m.%d %H:%M") + " -> " + self.__end_time.strftime("%Y.%m.%d %H:%M") + "  average power: " + str(self.average) + " num probes: " + str(len(self.__power_values))


class PvPowerForecast:

    def __init__(self, station_id: str, train_dir: str = None):
        if train_dir is None:
            train_dir = site_data_dir("pv_forecast", appauthor=False)
        self.weather_forecast_service = WeatherStation(station_id)
        self.train_log = TrainSampleLog(train_dir)
        self.__train_value_recorder = ValueRecorder()
        self.__estimator = Estimator(CoreVectorizer())
        self.__date_last_retrain = datetime.now()
        self.__date_best_estimator_selected = datetime.now()
        self.__retrain()
        Thread(target=self.__update_with_best_estimator, daemon=True).start()

    def __update_with_best_estimator(self):
        estimator = Trainer().train(self.train_log)
        logging.info("update estimator with best estimator variant " + str(estimator))
        self.__estimator = estimator

    def add_current_power_reading(self, real_power: int):
        if self.__train_value_recorder.is_expired():
            try:
                if not self.__train_value_recorder.empty():
                    weather_sample = self.weather_forecast_service.forecast(self.__train_value_recorder.time)
                    if weather_sample is not None:
                        annotated_sample = LabelledWeatherForecast.create(weather_sample,
                                                                          self.__train_value_recorder.average,
                                                                          time=self.__train_value_recorder.time)
                        if self.__estimator.usable_as_train_sample(annotated_sample):
                            self.__on_new_train_sample(annotated_sample)
            finally:
                self.__train_value_recorder = ValueRecorder()
        self.__train_value_recorder.add(real_power)

    def __on_new_train_sample(self, annotated_sample: LabelledWeatherForecast):
        # add to train log db
        self.train_log.append(annotated_sample)

        # retrain, if necessary
        max_retrain_period = 24*60 if (self.__estimator.num_samples_last_train < 5000) else 30*24*60  # 1 day or 1 month
        if datetime.now() > (self.__date_last_retrain + timedelta(minutes=max_retrain_period)):
            self.__date_last_retrain = datetime.now()
            if datetime.now() > (self.__date_best_estimator_selected + timedelta(days=15)):
                self.__date_best_estimator_selected = datetime.now()
                Thread(target=self.__update_with_best_estimator, daemon=True).start()
            else:
                Thread(target=self.__retrain, daemon=True).start()

    def __retrain(self):
        try:
            logging.info("retrain with train log " + self.train_log.filename())
            samples = self.train_log.all()
            train_report = self.__estimator.retrain(samples)
            logging.info("prediction model retrained: " + str(self.__estimator))
        except Exception as e:
            logging.warning("error occurred retrain prediction model " + str(e))

    def predict_by_weather_forecast(self, sample: WeatherForecast) -> int:
        return self.__estimator.predict(sample)

    def predict(self, time: datetime) -> Optional[int]:
        sample = self.weather_forecast_service.forecast(time)
        if sample is None:
            logging.info("could not predict power. Reason: no weather forcast data available (requested date time: " + time.strftime("%Y.%m.%d %H:%M") +
                         ". Available date time range: " + self.weather_forecast_service.forcast_from().strftime("%Y.%m.%d %H:%M") +
                         " -> " + self.weather_forecast_service.forcast_to().strftime("%Y.%m.%d %H:%M")  + "). Returning None")
            return None
        else:
            logging.debug("weather forecast for requested date time " + time.strftime("%Y.%m.%d %H:%M") + " ->  " + str(sample))
            predicted_power_watt = self.predict_by_weather_forecast(sample)
            logging.debug("predicted power " + str(predicted_power_watt) + " watt")
            return predicted_power_watt