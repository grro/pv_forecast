import logging
import pytz
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

    def __str__(self):
        return self.time.astimezone(pytz.UTC).strftime("%Y.%m.%d %H:%M") + " utc" + \
               ", irradiance=" + str(round(self.irradiance)) + \
               ", sunshine=" + str(round(self.sunshine)) + \
               ", cloud_cover=" + str(round(self.cloud_cover)) + \
               ", probability_for_fog=" + str(round(self.probability_for_fog)) + \
               ", visibility=" + str(round(self.visibility))


class WeatherStation:

    def __init__(self, station: str):
        self.__station = station
        self.__mosmixs = MosmixS.load(self.__station)
        self.__previous_mosmixs = self.__mosmixs

    def __refresh(self):
        mosmixs = MosmixS.load(self.__station)
        if mosmixs.utc_date_from() > self.__mosmixs.utc_date_from():
            logging.info("updated mosmix file loaded ")
            self.__previous_mosmixs = self.__mosmixs
            self.__mosmixs = mosmixs
        else:
            logging.info("loaded mosmix file is unchanged. ignoring it")

    def forcast_from(self) -> datetime:
        return self.__previous_mosmixs.data_from()

    def forcast_to(self) -> datetime:
        return self.__mosmixs.data_to()

    def __mosmix_is_expired(self) -> bool:
        return self.__mosmixs.content_age_sec() > (90*60) and self.__mosmixs.elapsed_sec_fetched() > (15*60)

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
                                   int(mosmixs.rad1h(time)),
                                   int(mosmixs.sund1(time)),
                                   int(mosmixs.neff(time)),
                                   int(mosmixs.wwm(time)),
                                   int(mosmixs.vv(time)))
        if forecast.is_valid():
            return forecast
        else:
            logging.info("available weather sample is incomplete. Returning None " + str(forecast))
            return None

