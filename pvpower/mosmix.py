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
from datetime import datetime, timedelta
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

    def merge(self, old_mosmix, min_local_datetime: datetime = None):
        if min_local_datetime is None:
            min_local_datetime = datetime.now() - timedelta(days=5)
        if old_mosmix is None:
            return self
        else:
            merged = MosmixS(self.station_id,
                             self.__issue_time_utc,
                             old_mosmix.__date_from_utc,
                             self.__date_to_utc,
                             {parameter: self.__parameter_series[parameter].merge(old_mosmix.__parameter_series[parameter], min_local_datetime) for parameter in self.__parameter_series.keys()})
            logging.debug("merging \nold mosmix:    " + str(old_mosmix) + " \nnew mosmix:    " + str(self) + " \nmerged mosmix: " + str(merged))
            return merged

    def is_expired(self) -> bool:
        content_age_min = int((datetime.now() - self.issue_time).total_seconds() / 60)
        return content_age_min > (60 + 25 + randrange(15))

    def supports(self, dt: datetime) -> bool:
        return self.date_from <= dt <= self.date_to

    def rad1h(self, dt: datetime) -> float:
        return self.__parameter_series["Rad1h"].value_at(dt)

    def sund1(self, dt: datetime) -> float:
        return self.__parameter_series["SunD1"].value_at(dt)

    def neff(self, dt: datetime) -> float:
        return self.__parameter_series["Neff"].value_at(dt)

    def wwm(self, dt: datetime) -> float:
        return self.__parameter_series["wwM"].value_at(dt)

    def vv(self, dt: datetime) -> float:
        return self.__parameter_series["VV"].value_at(dt)

    def __str__(self):
        return str(self.__parameter_series["Rad1h"].size()) + " entries (issued=" + self.issue_time.strftime("%Y.%m.%dT%H:%M") + "; " + self.date_from.strftime("%Y.%m.%dT%H:%M") + " -> " + self.date_to.strftime("%Y.%m.%dT%H:%M") + ")"

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



class MosmixLoader(ABC):

    @abstractmethod
    def get(self) -> MosmixS:
        pass


class MosmixWebLoader:

    def __init__(self, station_id: str):
        self.__station_id = station_id

    @staticmethod
    def __perform_get_chunked(url: str):
        with httpx.stream('GET', url) as r:
            yield from r.iter_bytes(chunk_size=65536)

    def get(self) -> MosmixS:
        url = 'https://opendata.dwd.de/weather/local_forecasts/mos/MOSMIX_S/all_stations/kml/MOSMIX_S_LATEST_240.kmz'

        issue_time_collector = IssueTimeCollector()
        timesteps_collector = TimeStepsCollector()
        forecasts_collector = ForecastValuesCollector(self.__station_id)

        xml_parser = ET.XMLPullParser(['start', 'end'])
        for file_name, file_size, unzipped_chunks in stream_unzip(MosmixWebLoader.__perform_get_chunked(url)):
            for chunk in unzipped_chunks:
                xml_parser.feed(chunk)
                for event, elem in xml_parser.read_events():
                    issue_time_collector.consume(event, elem)
                    timesteps_collector.consume(event, elem)
                    forecasts_collector.consume(event, elem)
        mosmix = MosmixS.create(self.__station_id,
                                issue_time_collector.issue_time_utc,
                                timesteps_collector.utc_timesteps,
                                forecasts_collector.parameters)
        logging.info("MosmixS file fetched " + str(mosmix))
        return mosmix



class FileCachedMosmixLoader(MosmixLoader):

    def __init__(self, station_id: str):
        self.__station_id = station_id
        self.__parent_loader = MosmixWebLoader(self.__station_id)
        self.__dir = site_data_dir("pv_forecast", appauthor=False)
        if not exists(self.__dir):
            os.makedirs(self.__dir)

    def get(self) -> MosmixS:
        cache_filename = os.path.join(self.__dir, "mosmixs_" + self.__station_id + ".json")
        cached_mosmix = MosmixS.load(cache_filename)
        if cached_mosmix is not None:
            if cached_mosmix.is_expired():
                elasped_minutes_since_last_cache_refresh = int((time.time() - os.path.getmtime(cache_filename)) / 60)
                if elasped_minutes_since_last_cache_refresh < 10:   # at maximum all 10 min the (large!) mosmix file will be loaded via web
                    logging.debug("filebased mosmix cache is expired, however last refresh is < 10 min. return cached one -> " + str(cached_mosmix))
                    return cached_mosmix
            else:
                return cached_mosmix
        mosmix = self.__parent_loader.get()
        merged_mosmix = mosmix.merge(cached_mosmix)
        merged_mosmix.save(cache_filename)
        return merged_mosmix


class MemoryCachedMosmixLoader(MosmixLoader):

    def __init__(self, station_id: str):
        self.__parent_loader = FileCachedMosmixLoader(station_id)
        self.__cached_mosmix = None

    def get(self) -> MosmixS:
        if self.__cached_mosmix is None or self.__cached_mosmix.is_expired():
            mosmix = self.__parent_loader.get()
            self.__cached_mosmix = mosmix
        return self.__cached_mosmix


