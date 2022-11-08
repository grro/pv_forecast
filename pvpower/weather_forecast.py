import logging
import pytz
from datetime import datetime
from pvpower.mosmix import MosmixSWeb
from typing import Optional


class WeatherForecast:

    def __init__(self,
                 time: datetime,
                 irradiance: int,
                 sunshine: int,
                 cloud_cover: int,
                 probability_for_fog: int,
                 visibility: int):
        self.time_utc = time.astimezone(pytz.UTC)
        self.irradiance = irradiance
        self.sunshine = sunshine
        self.cloud_cover = cloud_cover
        self.probability_for_fog = probability_for_fog
        self.visibility = visibility

    @property
    def time(self) -> datetime:
        return datetime.strptime((self.time_utc + (datetime.now() - datetime.utcnow())).strftime("%d.%m.%Y %H:%M:%S"), "%d.%m.%Y %H:%M:%S")

    def with_time(self, dt: datetime):
        return WeatherForecast(dt,
                               self.irradiance,
                               self.sunshine,
                               self.cloud_cover,
                               self.probability_for_fog,
                               self.visibility)

    def __str__(self):
        return self.time.strftime("%Y.%m.%d %H:%M") + \
               ", irradiance=" + str(round(self.irradiance)) + \
               ", sunshine=" + str(round(self.sunshine)) + \
               ", cloud_cover=" + str(round(self.cloud_cover)) + \
               ", probability_for_fog=" + str(round(self.probability_for_fog)) + \
               ", visibility=" + str(round(self.visibility))


class WeatherStation:

    def __init__(self, station: str):
        self.__station = station
        self.__mosmix = MosmixSWeb.load(self.__station)

    def forcast_from(self) -> datetime:
        return self.__mosmix.date_from

    def forcast_to(self) -> datetime:
        return self.__mosmix.date_to

    def forecast(self, time: datetime = None) -> Optional[WeatherForecast]:
        time = time if time is not None else datetime.now()

        if self.__mosmix.is_expired():
            mosmix = MosmixSWeb.load(self.__station)
            if mosmix.date_from > self.__mosmix.date_from:
                self.__mosmix = mosmix
                logging.info("updated mosmix file loaded")

        if self.__mosmix.supports(time):
            forecast = WeatherForecast(time,
                                       round(self.__mosmix.rad1h(time)),
                                       round(self.__mosmix.sund1(time)),
                                       round(self.__mosmix.neff(time)),
                                       round(self.__mosmix.wwm(time)),
                                       round(self.__mosmix.vv(time)))
            return forecast
        else:
            logging.info("forecast record for " + time.strftime("%Y.%m.%d %H:%M") + " not available. Returning None")
            return None
