import logging
import pickle
from os import path
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

    def __init__(self, station_id: str, pv_forecast_dir: str = None, estimator: Estimator = None):
        if pv_forecast_dir is None:
            self.__pv_forecast_dir = site_data_dir("pv_forecast", appauthor=False)
        else:
            self.__pv_forecast_dir = pv_forecast_dir
        self.train_log = TrainSampleLog(self.__pv_forecast_dir)
        self.weather_forecast_service = WeatherStation(station_id)
        self.__train_value_recorder = ValueRecorder()
        if estimator is None:
            self.__estimator = self.__load_default_estimator()
        else:
            self.__estimator = estimator
        self.__date_last_retrain = self.__estimator.date_last_train
        self.__retrain_if_old(False)

    def __load_default_estimator(self):
        try:
            with open(path.join(self.__pv_forecast_dir, 'default_estimator.pickle'), 'rb') as file:
                estimator = pickle.load(file)
                logging.debug("default estimator " + str(estimator) + " loaded from pickle file")
        except Exception as e:
            estimator = Estimator()
            logging.debug("default estimator " + str(estimator) + " created")
        return  estimator

    def __store_default_estimator(self):
        try:
            with open(path.join(self.__pv_forecast_dir, 'default_estimator.pickle'), 'wb') as file:
                pickle.dump(self.__estimator, file)
        except Exception as e:
            pass

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
            self.__retrain_if_old()

    def __retrain_if_old(self, background: bool = True):
        if datetime.now() > (self.__date_last_retrain + timedelta(minutes=23*60)):  # each 25 hours
            self.__date_last_retrain = datetime.now()
            if background:
                Thread(target=self.__retrain, daemon=True).start()
            else:
                self.__retrain()

    def __retrain(self):
        try:
            self.__estimator.retrain(self.train_log.all())
            logging.info("estimator retrained " + str(self.__estimator))
            self.__store_default_estimator()
        except Exception as e:
            logging.warning("error occurred retrain prediction model " + str(e))


    def __str__(self):
        return str(self.__estimator)
