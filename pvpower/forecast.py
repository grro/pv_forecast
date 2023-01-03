import logging
from datetime import datetime, timedelta
from typing import Optional
from pvpower.weather_forecast import WeatherStation, WeatherForecast
from pvpower.traindata import LabelledWeatherForecast, TrainSampleLog
from pvpower.estimator import Estimator
from pvpower.refreshing_estimator import AutoRefreshingEstimator


class ValueRecorder:

    def __init__(self):
        self.start_time = datetime.strptime(datetime.now().strftime("%d.%m.%Y %H") + ":00", "%d.%m.%Y %H:%M")
        self.end_time = self.start_time + timedelta(minutes=60)
        self.__power_values = []
        #logging.debug("value recorder created (" + str(self) + ")")

    def empty(self) -> bool:
        return len(self.__power_values) == 0

    def is_expired(self):
        return datetime.now() >= self.end_time

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


class PvPowerForecast:

    def __init__(self, station_id: str, pv_forecast_dir: None, estimator: Estimator = None):
        self.__train_value_recorder = ValueRecorder()
        self.weather_forecast_service = WeatherStation(station_id)
        self.train_log = TrainSampleLog(pv_forecast_dir)
        self.__estimator = estimator if estimator is not None else AutoRefreshingEstimator(self.train_log)

    def add_current_power_reading(self, real_power: int):
        if self.__train_value_recorder.is_expired():
            try:
                if not self.__train_value_recorder.empty():
                    weather_sample = self.weather_forecast_service.forecast(self.__train_value_recorder.start_time)
                    if weather_sample is not None:
                        annotated_sample = LabelledWeatherForecast.create(weather_sample,
                                                                          self.__train_value_recorder.average,
                                                                          time=self.__train_value_recorder.start_time)
                        logging.info("add train sample " + str(annotated_sample))
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
        return self.__estimator.predict(sample)

    def __str__(self):
        return str(self.__estimator)
