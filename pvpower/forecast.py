import logging
from appdirs import site_data_dir
from threading import Thread
from datetime import datetime, timedelta
from typing import Optional
from pvpower.weather_forecast import WeatherStation, WeatherForecast
from pvpower.traindata import LabelledWeatherForecast, TrainSampleLog
from pvpower.estimator import Estimator



class ValueRecorder:

    def __init__(self, window_size_min: int = 10):
        self.__start_time = datetime.now()
        self.__end_time = datetime.now() + timedelta(minutes=window_size_min)
        self.time = self.__start_time + timedelta(minutes=round(window_size_min/2))
        self.__power_values = []

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
        logging.info("train log " + self.train_log.filename())
        self.__train_value_recorder = ValueRecorder()
        self.__estimator = Estimator()
        self.__date_last_retrain = datetime.now() - timedelta(days=90)
        self.__num_samples_last_retrain = 0
        self.__date_last_weather_forecast = datetime.now() - timedelta(days=1)
        self.__retrain()

    @property
    def __max_retrain_period_minutes(self) -> int:
        if self.__num_samples_last_retrain < 1000:
            return 60       # 1 hour
        else:
            return 7*24*60  # 1 week

    def __retrain(self):
        try:
            if datetime.now() > (self.__date_last_retrain + timedelta(minutes=self.__max_retrain_period_minutes)):
                samples = self.train_log.all()
                self.__estimator.retrain(samples)
                self.__date_last_retrain = datetime.now()
                self.__num_samples_last_retrain = len(samples)
                logging.info("prediction model retrained: " + str(self.__estimator))

        except Exception as e:
            logging.warning("error occurred retrain prediction model " + str(e))

    def current_power_reading(self, real_power: int):
        if self.__train_value_recorder.is_expired():
            try:
                if self.__train_value_recorder.average is not None:
                    weather_sample = self.weather_forecast_service.forecast(self.__train_value_recorder.time)
                    if weather_sample is not None:
                        annotated_sample = LabelledWeatherForecast.create(weather_sample,
                                                                          self.__train_value_recorder.average,
                                                                          time=self.__train_value_recorder.time)
                        self.train_log.append(annotated_sample)
            finally:
                self.__train_value_recorder = ValueRecorder()
            Thread(target=self.__retrain, daemon=True).start()
        self.__train_value_recorder.add(real_power)

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
            logging.debug("got weather forecast for requested date time: " + time.strftime("%Y.%m.%d %H:%M") + ": " + str(sample))
            predicted_power_watt = self.predict_by_weather_forecast(sample)
            logging.debug("predicted power " + str(predicted_power_watt) + " watt")
            return predicted_power_watt