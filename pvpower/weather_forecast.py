import logging
from dataclasses import dataclass
from datetime import datetime
from pvpower.mosmix import MosmixS
from typing import Optional


@dataclass(frozen=True)
class WeatherForecast:
    time: datetime
    irradiance: int
    sunshine: int
    cloud_cover: int
    probability_for_fog: int
    visibility: int

    def with_time(self, dt: datetime):
        return WeatherForecast(dt,
                               self.irradiance,
                               self.sunshine,
                               self.cloud_cover,
                               self.probability_for_fog,
                               self.visibility)

    def is_valid(self):
        return self.time is not None and \
               self.irradiance is not None and \
               self.sunshine is not None and \
               self.cloud_cover is not None and \
               self.probability_for_fog is not None and \
               self.visibility is not None

class WeatherStation:

    def __init__(self, station: str):
        self.__station = station
        self.__mosmixs = MosmixS.load(self.__station)
        self.__pervious_mosmixs = self.__mosmixs

    def __refresh(self):
        mosmixs = MosmixS.load(self.__station)
        if mosmixs.start_date() > self.__mosmixs.start_date():   # save old prediction
            self.__pervious_mosmixs = self.__mosmixs
        self.__mosmixs = mosmixs

    def forcast_from(self) -> datetime:
        return self.__pervious_mosmixs.data_from()

    def forcast_to(self) -> datetime:
        return self.__mosmixs.data_to()

    def forecast(self, time: datetime = None) -> Optional[WeatherForecast]:
        if self.__mosmixs.is_expired():
            self.__refresh()

        time = time if time is not None else datetime.now()

        if self.__mosmixs.supports(time):
            mosmixs = self.__mosmixs
        elif self.__pervious_mosmixs.supports(time):
            logging.info("fallback")
            mosmixs = self.__pervious_mosmixs
        else:
            return None

        forecast = WeatherForecast(time,
                                   mosmixs.rad1h(time),
                                   mosmixs.sund1(time),
                                   mosmixs.neff(time),
                                   mosmixs.wwm(time),
                                   mosmixs.vv(time))
        if forecast.is_valid():
            return forecast
        else:
            logging.info("weather sample is incomplete. Returning None " + str(forecast))
            return None

