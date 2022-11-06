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


class Trainer:

    def select_best_estimator(self, train_log: TrainSampleLog, num_rounds: int = 5) -> Estimator:
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

        days = len(set([sample.time.strftime("%Y.%m.%d") for sample in samples]))
        report = "tested with " + str(len(samples)) + " cleaned samples (" + str(days) + " days; " + str(num_rounds) + " test rounds per variant)" + "\n"
        report += "VARIANT ............................ DERIVATION\n"
        for variant, vectorizer in vectorizer_map.items():
            estimator = Estimator(vectorizer)
            median_report = estimator.test(samples, rounds=num_rounds)
            score_str = str(round(median_report.score, 1))
            report += variant + " " + "".join(["."] * (45 - (len(variant)+len(score_str)))) + " " + score_str + "\n"
            if median_report.score < lowest_score:
                best_estimator = estimator
                lowest_score = median_report.score
        logging.info(report)
        best_estimator.retrain(samples)
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
        self.__date_last_retrain = datetime.now() - timedelta(days=90)
        self.__date_best_estimator_selected = datetime.now() - timedelta(days=90)
        self.__estimator = Estimator(CoreVectorizer())
        self.__retrain()

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
                            self.train_log.append(annotated_sample)
            finally:
                self.__train_value_recorder = ValueRecorder()
        self.__train_value_recorder.add(real_power)

    def predict(self, time: datetime) -> Optional[int]:
        sample = self.weather_forecast_service.forecast(time)
        if sample is None:
            logging.info("could not predict power. Reason: no weather forcast data available (requested date time: " + time.strftime("%Y.%m.%d %H:%M") +
                         ". Available date time range: " + self.weather_forecast_service.forcast_from().strftime("%Y.%m.%d %H:%M") +
                         " -> " + self.weather_forecast_service.forcast_to().strftime("%Y.%m.%d %H:%M")  + "). Returning None")
            return None
        else:
            return self.predict_by_weather_forecast(sample)

    def predict_by_weather_forecast(self, sample: WeatherForecast) -> int:
        try:
            return self.__estimator.predict(sample)
        finally:
            # retrain, if necessary
            if datetime.now() > (self.__date_last_retrain + timedelta(minutes=23*60)):  # each 25 hours
                self.__date_last_retrain = datetime.now()
                if datetime.now() > (self.__date_best_estimator_selected + timedelta(days=30)):  # each 1 month
                    self.__date_best_estimator_selected = datetime.now()
                    Thread(target=self.__update_with_best_estimator, daemon=True).start()
                else:
                    Thread(target=self.__retrain, daemon=True).start()

    def __retrain(self):
        try:
            self.__estimator.retrain(self.train_log.all())
            logging.info("estimator retrained " + str(self.__estimator))
        except Exception as e:
            logging.warning("error occurred retrain prediction model " + str(e))

    def __update_with_best_estimator(self):
        self.__estimator = Trainer().select_best_estimator(self.train_log)
        logging.info("update estimator with best estimator variant " + str(self.__estimator))
