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
        self.__previous_mosmixs = self.__mosmixs

    def __refresh(self):
        mosmixs = MosmixS.load(self.__station)
        if mosmixs.start_date() > self.__mosmixs.start_date():   # save old prediction
            self.__previous_mosmixs = self.__mosmixs
        self.__mosmixs = mosmixs

    def forcast_from(self) -> datetime:
        return self.__previous_mosmixs.data_from()

    def forcast_to(self) -> datetime:
        return self.__mosmixs.data_to()

    def __mosmix_is_expired(self) -> bool:
        return self.__mosmixs.content_age_sec() > (60*60) and self.__mosmixs.elapsed_sec_fetched() > (25*60)

    def forecast(self, time: datetime = None) -> Optional[WeatherForecast]:
        time = time if time is not None else datetime.now()

        if self.__mosmix_is_expired():
            self.__refresh()

        if self.__mosmixs.supports(time):
            mosmixs = self.__mosmixs
        elif self.__previous_mosmixs.supports(time):
            mosmixs = self.__previous_mosmixs
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
            logging.info("available weather sample is incomplete. Returning None " + str(forecast))
            return None

