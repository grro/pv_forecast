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
        self.utc_time = time.astimezone(pytz.UTC)
        self.irradiance = irradiance
        self.sunshine = sunshine
        self.cloud_cover = cloud_cover
        self.probability_for_fog = probability_for_fog
        self.visibility = visibility

    def with_time(self, dt: datetime):
        return WeatherForecast(dt,
                               self.irradiance,
                               self.sunshine,
                               self.cloud_cover,
                               self.probability_for_fog,
                               self.visibility)

    def is_valid(self):
        return self.utc_time is not None and \
               self.irradiance is not None and \
               self.sunshine is not None and \
               self.cloud_cover is not None and \
               self.probability_for_fog is not None and \
               self.visibility is not None

    def __str__(self):
        return self.utc_time.strftime("%Y.%m.%d %H:%M") + " utc" + \
               ", irradiance=" + str(round(self.irradiance)) + \
               ", sunshine=" + str(round(self.sunshine)) + \
               ", cloud_cover=" + str(round(self.cloud_cover)) + \
               ", probability_for_fog=" + str(round(self.probability_for_fog)) + \
               ", visibility=" + str(round(self.visibility))


class WeatherStation:

    def __init__(self, station: str, ):
        self.__station = station
        self.__mosmix = MosmixSWeb.load(self.__station)

    def forcast_from(self) -> datetime:
        return self.__mosmix.utc_date_from

    def forcast_to(self) -> datetime:
        return self.__mosmix.utc_date_to

    def forecast(self, time: datetime = None) -> Optional[WeatherForecast]:
        time = time if time is not None else datetime.now()

        if self.__mosmix.is_expired():
            mosmix = MosmixSWeb.load(self.__station)
            if mosmix.utc_date_from > self.__mosmix.utc_date_from:
                logging.info("updated mosmix file loaded")
                self.__mosmix = mosmix

        if self.__mosmix.supports(time):
            forecast = WeatherForecast(time,
                                       int(self.__mosmix.rad1h(time)),
                                       int(self.__mosmix.sund1(time)),
                                       int(self.__mosmix.neff(time)),
                                       int(self.__mosmix.wwm(time)),
                                       int(self.__mosmix.vv(time)))
            if forecast.is_valid():
                return forecast
            else:
                logging.info("available weather reacord is incomplete. Returning None " + str(forecast))
                return None
        else:
            logging.info("forecast record for " + time.strftime("%Y.%m.%d %H:%M") + " not available. Returning None")
            return None
