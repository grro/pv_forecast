import logging
import pickle
from os import path, makedirs
from os.path import exists
from appdirs import user_cache_dir, site_data_dir
from threading import Thread
from datetime import datetime, timedelta
from typing import Optional
from pvpower.weather_forecast import WeatherStation, WeatherForecast
from pvpower.traindata import LabelledWeatherForecast, TrainSampleLog, TrainData
from pvpower.estimator import Estimator, SVMEstimator, TrainReport, FullVectorizer


class ValueRecorder:

    def __init__(self):
        self.start_time = datetime.strptime(datetime.now().strftime("%d.%m.%Y %H") + ":00", "%d.%m.%Y %H:%M")
        self.end_time = self.start_time + timedelta(minutes=60)
        self.__power_values = []
        #logging.debug("value recorder created (" + str(self) + ")")

    def empty(self) -> bool:
        return len(self.__power_values) == 0

    def is_expired(self):
        return datetime.now() > self.end_time

    def add(self, value: int):
        self.__power_values.append(value)
        #logging.debug("record added to value recorder (" + str(self) + ")")

    @property
    def average(self) -> Optional[int]:
        if len(self.__power_values) == 0:
            return None
        else:
            return int(round(sum(self.__power_values) / len(self.__power_values), 0))

    def __str__(self):
        return self.start_time.strftime("%Y.%m.%d %H:%M") + " -> " + self.end_time.strftime("%Y.%m.%d %H:%M") + "  average power: " + str(self.average) + " num probes: " + str(len(self.__power_values))



class PersistentEstimator(Estimator):

    def __init__(self, default_estimator: Estimator):
        cache_dir = user_cache_dir("pv_forecast", appauthor=False)
        if not path.exists(cache_dir):
            makedirs(cache_dir)
        self.__filename = path.join(cache_dir, 'Estimator_' + default_estimator.variant + '.p')
        self.__estimator = default_estimator
        if exists(self.__filename):
            try:
                with open(self.__filename, 'rb') as file:
                    self.__estimator = pickle.load(file)
                    logging.debug("estimator " + str(self.__estimator) + " loaded from pickle file (" + self.__filename + ")")
            except Exception as e:
                logging.warning("error occurred loading estimator " + str(e))

    @property
    def variant(self) -> str:
        return self.__estimator.variant

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
            with open(self.__filename, 'wb') as file:
                pickle.dump(self.__estimator, file)
        except Exception as e:
            pass

    def __str__(self):
        return str(self.__estimator)


class PvPowerForecast:

    def __init__(self, station_id: str, pv_forecast_dir: None, estimator: Estimator = None):
        pv_forecast_dir = pv_forecast_dir if pv_forecast_dir is not None else site_data_dir("pv_forecast", appauthor=False)
        self.train_log = TrainSampleLog(pv_forecast_dir)
        self.__train_value_recorder = ValueRecorder()
        self.weather_forecast_service = WeatherStation(station_id)
        self.__estimator = estimator if estimator is not None else PersistentEstimator(SVMEstimator(FullVectorizer()))
        self.__date_last_retrain_initiated = self.__estimator.date_last_train()
        self.__train_if_old(False)

    def add_current_power_reading(self, real_power: int):
        if self.__train_value_recorder.is_expired():
            try:
                if not self.__train_value_recorder.empty():
                    weather_sample = self.weather_forecast_service.forecast(self.__train_value_recorder.start_time)
                    if weather_sample is not None:
                        annotated_sample = LabelledWeatherForecast.create(weather_sample,
                                                                          self.__train_value_recorder.average,
                                                                          time=self.__train_value_recorder.start_time)
                        self.train_log.append(annotated_sample)
            finally:
                self.__train_value_recorder = ValueRecorder()
        self.__train_value_recorder.add(real_power)

    def predict(self, time: datetime) -> Optional[int]:
        sample = self.weather_forecast_service.forecast(time)
        if sample is None:
            logging.info("could not predict power. Reason: no weather forecast data available (requested date time: " + time.strftime("%Y.%m.%d %H:%M") +
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
        if datetime.now() > (self.__date_last_retrain_initiated + timedelta(minutes=3 * 60)):
            self.__date_last_retrain_initiated = datetime.now()
            if background:
                Thread(target=self.__estimator.retrain, daemon=True, args=(self.train_log.all(),)).start()
            else:
                self.__estimator.retrain(self.train_log.all())

    def __str__(self):
        return str(self.__estimator)
