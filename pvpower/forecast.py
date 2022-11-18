import logging
import pickle
from os import path, makedirs
from appdirs import site_data_dir, user_cache_dir
from threading import Thread
from datetime import datetime, timedelta
from typing import Optional
from pvpower.weather_forecast import WeatherStation, WeatherForecast
from pvpower.traindata import LabelledWeatherForecast, TrainSampleLog, TrainData
from pvpower.estimator import Estimator, SMVEstimator, TrainReport


class ValueRecorder:

    def __init__(self, window_size_min: int = 20):
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



class DefaultEstimator(Estimator):

    def __init__(self, estimator: Estimator, pv_forecast_dir: str):
        self.__estimator = estimator
        self.__pv_forecast_dir = pv_forecast_dir

    @staticmethod
    def __filename() -> str:
        dir = user_cache_dir("pv_forecast", appauthor=False)
        if not path.exists(dir):
            makedirs(dir)
        return path.join(dir, 'dflt_estimator.pickle')

    def date_last_train(self) -> datetime:
        return self.__estimator.date_last_train()

    def predict(self, sample: WeatherForecast) -> int:
        return self.__estimator.predict(sample)

    def retrain(self, train_data: TrainData) -> TrainReport:
        train_report = self.__estimator.retrain(train_data)
        self.__store()
        return train_report

    def __store(self):
        try:
            with open(self.__filename(), 'wb') as file:
                pickle.dump(self.__estimator, file)
        except Exception as e:
            pass

    @staticmethod
    def get(pv_forecast_dir: str):
        try:
            with open(DefaultEstimator.__filename(), 'rb') as file:
                estimator = DefaultEstimator(pickle.load(file), pv_forecast_dir)
                estimator.retrain(TrainSampleLog(pv_forecast_dir).all())
                logging.debug("default estimator " + str(estimator) + " loaded from pickle file (" + DefaultEstimator.__filename() + ") and retrained")
        except Exception as e:
            estimator = DefaultEstimator(SMVEstimator(), pv_forecast_dir)
            estimator.retrain(TrainSampleLog(pv_forecast_dir).all())
            logging.debug("default estimator " + str(estimator) + " created and trained")
        return estimator

    def __str__(self):
        return str(self.__estimator)

class PvPowerForecast:

    def __init__(self, station_id: str, pv_forecast_dir: str = None, estimator: Estimator = None):
        if pv_forecast_dir is None:
            pv_forecast_dir = site_data_dir("pv_forecast", appauthor=False)
        self.train_log = TrainSampleLog(pv_forecast_dir)
        self.weather_forecast_service = WeatherStation(station_id)
        self.__train_value_recorder = ValueRecorder()
        if estimator is None:
            self.__estimator = DefaultEstimator.get(pv_forecast_dir)
        else:
            self.__estimator = estimator
        self.__date_last_retrain = self.__estimator.date_last_train()
        self.__train_if_old(False)

    def add_current_power_reading(self, real_power: int):
        if self.__train_value_recorder.is_expired():
            try:
                if not self.__train_value_recorder.empty():
                    weather_sample = self.weather_forecast_service.forecast(self.__train_value_recorder.time)
                    if weather_sample is not None:
                        annotated_sample = LabelledWeatherForecast.create(weather_sample,
                                                                          self.__train_value_recorder.average,
                                                                          time=self.__train_value_recorder.time)
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
            self.__train_if_old()

    def __train_if_old(self, background: bool = True):
        if datetime.now() > (self.__date_last_retrain + timedelta(minutes=3*60)):
            self.__date_last_retrain = datetime.now()
            if background:
                Thread(target=self.__estimator.retrain, daemon=True, args=(self.train_log.all(),)).start()
            else:
                self.__estimator.retrain(self.train_log.all())

    def __str__(self):
        return str(self.__estimator)
