import logging
import os.path
import httpx
import json
import time
from abc import ABC, abstractmethod
from appdirs import site_data_dir
from os.path import exists
from stream_unzip import stream_unzip
from random import randrange
import pytz
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional
import xml.etree.ElementTree as ET



class IssueTimeCollector:

    def __init__(self):
        self.issue_time_utc = None

    def consume(self, event, elem):
        if elem.tag.endswith("IssueTime"):
            if elem.text is not None:
                self.issue_time_utc = datetime.fromisoformat(elem.text.replace("Z", "+00:00"))


class TimeStepsCollector:

    def __init__(self):
        self.__is_collecting = False
        self.utc_timesteps = []

    def consume(self, event, elem):
        if event == 'start' and elem.tag.endswith("ForecastTimeSteps"):
            self.utc_timesteps.clear()
            self.__is_collecting = True
        elif event == 'start' and self.__is_collecting and elem.tag.endswith("TimeStep"):
            if elem.text is not None:
                utc_time = datetime.fromisoformat(elem.text.replace("Z", "+00:00"))
                self.utc_timesteps.append(utc_time)
        elif self.__is_collecting and event == 'end' and elem.tag.endswith("ForecastTimeSteps"):
            self.__is_collecting = False


class ForecastValuesCollector:

    def __init__(self, station_id: str):
        self.station_id = station_id
        self.__parameter_name = ""
        self.parameters: Dict[str, List[float]] = {}
        self.__is_placemark_collecting = False
        self.__is_forecast_collecting = False

    def consume(self, event, elem):
        if event == 'start' and elem.tag.endswith("Placemark"):
            self.__is_placemark_collecting = True

        elif self.__is_placemark_collecting and elem.tag.endswith("name"):
            if elem.text is not None:
                if self.station_id == elem.text:
                    self.__is_forecast_collecting = True
                    self.parameters.clear()
                else:
                    self.__is_forecast_collecting = False

        elif self.__is_forecast_collecting and event == 'start' and elem.tag.endswith("Forecast"):
            for name in elem.attrib.keys():
                if name.endswith("elementName"):
                    self.__parameter_name = elem.attrib[name]
                    break

        elif self.__is_forecast_collecting and elem.tag.endswith("value"):
            if elem.text is not None:
                values = elem.text.split()
                self.parameters[self.__parameter_name] = [(None if value == '-' else float(value)) for value in values]

        elif event == 'end' and elem.tag.endswith("Placemark"):
            self.__is_placemark_collecting = False


class ParameterUtcSeries:

    def __init__(self, name: str, series: Dict[str, float]):
        self.name = name
        self.__series = series   # time_utc: value

    def size(self) -> int:
        return len(self.__series.keys())

    def value_at(self, local_datetime: datetime) -> float:
        dt_utc = local_datetime.astimezone(pytz.UTC)
        return self.__series.get(dt_utc.strftime("%Y.%m.%d %H"))

    def to_dict(self) -> Dict[str, Any]:
        return { "name": self.name,
                 "series": self.__series }

    def merge(self, other, min_local_datetime: datetime):
        min_datetime_utc = min_local_datetime.astimezone(pytz.UTC)
        min_datetime_utc_str = min_datetime_utc.strftime("%Y.%m.%d %H")
        merged_series = {}
        for time_utc in other.__series.keys():
            if time_utc >= min_datetime_utc_str:
                merged_series[time_utc] = other.__series[time_utc]
        merged_series.update(self.__series)
        return ParameterUtcSeries(self.name, merged_series)

    @staticmethod
    def create(name: str, time_series: List[datetime], values: Dict[str, List[float]]):
        return ParameterUtcSeries(name, {time_series[i].astimezone(pytz.UTC).strftime("%Y.%m.%d %H"): values[name][i] for i in range(0, len(time_series))})

    @staticmethod
    def from_dict(map: Dict[str, Any]):
        return ParameterUtcSeries(map['name'], map['series'])


class MosmixS:

    @staticmethod
    def create(station_id: str,
               issue_time_utc: datetime,
               timesteps_utc: List[datetime],
               parameters: Dict[str, List[float]]):
        return MosmixS(station_id,
                       issue_time_utc,
                       timesteps_utc[0],
                       timesteps_utc[-1],
                       {parameter: ParameterUtcSeries.create(parameter, timesteps_utc, parameters) for parameter in parameters})

    def __utc_to_local(utc: datetime) -> datetime:
        return datetime.strptime((utc + (datetime.now() - datetime.utcnow())).strftime("%d.%m.%Y %H:%M:%S.%f"), "%d.%m.%Y %H:%M:%S.%f")

    def __init__(self,
                 station_id: str,
                 issue_time_utc: datetime,
                 date_from_utc: datetime,
                 date_to_utc: datetime,
                 parameter_series: Dict[str, ParameterUtcSeries]):
        self.station_id = station_id
        self.__issue_time_utc = issue_time_utc
        self.__date_from_utc = date_from_utc
        self.__date_to_utc = date_to_utc
        self.__parameter_series = parameter_series

    @property
    def issue_time(self) -> datetime:
        return MosmixS.__utc_to_local(self.__issue_time_utc)

    @property
    def date_from(self) -> datetime:
        return MosmixS.__utc_to_local(self.__date_from_utc)

    @property
    def date_to(self) -> datetime:
        return MosmixS.__utc_to_local(self.__date_to_utc)

    def merge(self, old_mosmix, min_local_datetime: datetime):
        if old_mosmix is None:
            return self
        else:
            merged = MosmixS(self.station_id,
                             self.__issue_time_utc,
                             old_mosmix.__date_from_utc,
                             self.__date_to_utc,
                             {parameter: self.__parameter_series[parameter].merge(old_mosmix.__parameter_series[parameter], min_local_datetime) for parameter in self.__parameter_series.keys()})
            logging.debug("merging \nold mosmix: " + str(old_mosmix) + " \nnew mosmix: " + str(self) + " \n-> " + str(merged))
            return merged

    def is_expired(self) -> bool:
        content_age_min = int((datetime.now(timezone.utc) - self.__issue_time_utc).total_seconds() / 60)
        return content_age_min > (60 + 25 + randrange(15))

    def supports(self, local_datetime: datetime) -> bool:
        dt_utc = local_datetime.astimezone(pytz.UTC)
        return self.__date_from_utc <= dt_utc <= self.__date_to_utc

    def rad1h(self, local_datetime: datetime) -> float:
        return self.__parameter_series["Rad1h"].value_at(local_datetime)

    def sund1(self, local_datetime: datetime) -> float:
        return self.__parameter_series["SunD1"].value_at(local_datetime)

    def neff(self, local_datetime: datetime) -> float:
        return self.__parameter_series["Neff"].value_at(local_datetime)

    def wwm(self, local_datetime: datetime) -> float:
        return self.__parameter_series["wwM"].value_at(local_datetime)

    def vv(self, local_datetime: datetime) -> float:
        return self.__parameter_series["VV"].value_at(local_datetime)

    def __str__(self):
        return "issued=" + self.issue_time.strftime("%d.%m.%Y %H:%M") + " / " + str(self.__parameter_series["Rad1h"].size()) + " entries (" + self.date_from.strftime("%d.%m.%Y %H:%M") + " -> " + self.date_to.strftime("%d.%m.%Y %H:%M") + ")"

    def save(self, filename: str = "mosmix.json"):
        with open(filename, "w") as file:
            data = json.dumps({ "station_id": self.station_id,
                                "issue_time_utc": self.__issue_time_utc.isoformat(),
                                "utc_date_from": self.__date_from_utc.isoformat(),
                                "utc_date_to": self.__date_to_utc.isoformat(),
                                "parameter_series": { parameter: self.__parameter_series[parameter].to_dict() for parameter in self.__parameter_series.keys()}})
            file.write(data)

    @staticmethod
    def load(filename: str = "mosmix.json"):
        if exists(filename):
            with open(filename, "r") as file:
                try:
                    data = json.loads(file.read())
                    station_id = data['station_id']
                    issue_time_utc = datetime.fromisoformat(data['issue_time_utc'])
                    utc_date_from = datetime.fromisoformat(data['utc_date_from'])
                    utc_date_to = datetime.fromisoformat(data['utc_date_to'])
                    parameter_series = {parameter: ParameterUtcSeries.from_dict(data['parameter_series'][parameter]) for parameter in data['parameter_series'].keys()}
                    return MosmixS(station_id,
                                   issue_time_utc,
                                   utc_date_from,
                                   utc_date_to,
                                   parameter_series)
                except Exception as e:
                    logging.warning("error occurred loading mosmix cache file " + filename, e)
        return None





class MosmixCache(ABC):

    @abstractmethod
    def put(self, station_id: str, mosmix: MosmixS):
        return None

    @abstractmethod
    def get(self, station_id: str) -> Optional[MosmixS]:
        return None


class MemoryBasedMosmixCache(MosmixCache):

    def __init__(self):
        self.__cached_mosmix_map = {}

    def put(self, station_id, mosmix: MosmixS):
        self.__cached_mosmix_map[station_id] = mosmix

    def get(self, station_id) -> Optional[MosmixS]:
        cached_mosmix = self.__cached_mosmix_map.get(station_id, None)
        if cached_mosmix is None or cached_mosmix.is_expired():
            return None
        else:
            return cached_mosmix


class FileBasedMosmixCache(MosmixCache):

    def __init__(self, ):
        self.__dir = site_data_dir("pv_forecast", appauthor=False)
        if not exists(self.__dir):
            os.makedirs(self.__dir)

    def __filename(self, station_id: str):
        return os.path.join(self.__dir, "mosmixs_" + station_id + ".json")

    def put(self, station_id, mosmix: MosmixS):
        cache_filename = self.__filename(station_id)
        mosmix.save(cache_filename)

    def get(self, station_id) -> Optional[MosmixS]:
        cache_filename = self.__filename(station_id)
        cached_mosmix = MosmixS.load(cache_filename)
        if cached_mosmix is not None:
            if cached_mosmix.is_expired():
                elasped_minutes_since_last_cache_refresh = int((time.time() - os.path.getmtime(cache_filename)) / 60)
                if elasped_minutes_since_last_cache_refresh < 10:   # at maximum all 10 min the (large!) mosmix file will be loaded via web
                    logging.debug("filebased mosmix cache is expired, however last refresh is < 10 min. return cached one")
                    return cached_mosmix
            else:
                return cached_mosmix
        return None


class TieredMosmixCache(ABC):

    def __init__(self):
        self.__in_memory_cache = MemoryBasedMosmixCache()
        self.__file_cache = FileBasedMosmixCache()

    def put(self, station_id: str, mosmix: MosmixS):
        self.__in_memory_cache.put(station_id, mosmix)
        self.__file_cache.put(station_id, mosmix)

    def get(self, station_id: str) -> Optional[MosmixS]:
        cached_mosmix = self.__in_memory_cache.get(station_id)
        if cached_mosmix is None or cached_mosmix.is_expired():
            logging.debug("in memory cache is expired, loading mosmix from filebased cache")
            return self.__file_cache.get(station_id)
        else:
            return cached_mosmix



class MosmixSWeb:

    __cache = TieredMosmixCache()

    # x-check https://mosmix.de/online.html#/station/10724/station
    # parameters (https://dwd-geoportal.de/products/G_FJM/)

    def __init__(self, station_id: str):
        self.station_id = station_id
        self.__issue_time_collector = IssueTimeCollector()
        self.__timesteps_collector = TimeStepsCollector()
        self.__forecasts_collector = ForecastValuesCollector(station_id)

    def consume(self, event, elem):
        self.__issue_time_collector.consume(event, elem)
        self.__timesteps_collector.consume(event, elem)
        self.__forecasts_collector.consume(event, elem)

    @staticmethod
    def __perform_get_chunked(url):
        with httpx.stream('GET', url) as r:
            yield from r.iter_bytes(chunk_size=65536)

    @staticmethod
    def __load_from_web(station_id: str) -> MosmixS:
        url = 'https://opendata.dwd.de/weather/local_forecasts/mos/MOSMIX_S/all_stations/kml/MOSMIX_S_LATEST_240.kmz'
        mosmix_loader = MosmixSWeb(station_id)
        xml_parser = ET.XMLPullParser(['start', 'end'])
        for file_name, file_size, unzipped_chunks in stream_unzip(MosmixSWeb.__perform_get_chunked(url)):
            for chunk in unzipped_chunks:
                xml_parser.feed(chunk)
                for event, elem in xml_parser.read_events():
                    mosmix_loader.consume(event, elem)
        return MosmixS.create(station_id,
                              mosmix_loader.__issue_time_collector.issue_time_utc,
                              mosmix_loader.__timesteps_collector.utc_timesteps,
                              mosmix_loader.__forecasts_collector.parameters)

    @staticmethod
    def load(station_id: str) -> MosmixS:
        # load cached mosmix
        cached_mosmix = MosmixSWeb.__cache.get(station_id)
        if cached_mosmix is not None:
            return cached_mosmix
        else:   # no valid cache entry
            mosmix = MosmixSWeb.__load_from_web(station_id)
            mosmix = mosmix.merge(cached_mosmix, datetime.now() - timedelta(days=1))
            MosmixSWeb.__cache.put(station_id, mosmix)
            return mosmix
