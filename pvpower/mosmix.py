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
from typing import List, Dict
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

    TIME_PATTERN = "%Y.%m.%dT%H"

    @staticmethod
    def __datetime_to_utc_string(dt: datetime) -> str:
        return dt.astimezone(pytz.UTC).strftime(ParameterUtcSeries.TIME_PATTERN)

    @staticmethod
    def __utc_string_to_datetime(utc_string: str) -> datetime:
        return datetime.strptime(utc_string, ParameterUtcSeries.TIME_PATTERN)

    @staticmethod
    def create(name: str, time_series: List[datetime], values: Dict[str, List[float]]):
        return ParameterUtcSeries({ParameterUtcSeries.__datetime_to_utc_string(time_series[i]): values[name][i] for i in range(0, len(time_series))})

    def __init__(self, utc_series: Dict[str, float]):
        self.utc_series = utc_series

    def size(self) -> int:
        return len(self.utc_series.keys())

    @property
    def date_from_utc(self) -> datetime:
        return self.__utc_string_to_datetime(sorted(self.utc_series.keys())[0])

    @property
    def date_to_utc(self) -> datetime:
        return self.__utc_string_to_datetime(sorted(self.utc_series.keys())[-1])

    def value_at(self, dt: datetime) -> float:
        return self.utc_series.get(self.__datetime_to_utc_string(dt))

    def merge(self, old, min_datetime: datetime):
        series = {time_utc: old.utc_series[time_utc] for time_utc in old.utc_series.keys() if time_utc >= self.__datetime_to_utc_string(min_datetime)}
        series.update(self.utc_series)
        return ParameterUtcSeries(series)


class MosmixS:

    @staticmethod
    def create(station_id: str,
               issue_time_utc: datetime,
               timesteps_utc: List[datetime],
               parameters: Dict[str, List[float]]):
        return MosmixS(station_id,
                       issue_time_utc,
                       {parameter: ParameterUtcSeries.create(parameter, timesteps_utc, parameters) for parameter in parameters})

    def __utc_to_local(utc: datetime) -> datetime:
        return datetime.strptime((utc + (datetime.now() - datetime.utcnow())).strftime("%d.%m.%Y %H:%M:%S.%f"), "%d.%m.%Y %H:%M:%S.%f")

    def __init__(self,
                 station_id: str,
                 issue_time_utc: datetime,
                 parameter_series: Dict[str, ParameterUtcSeries]):
        self.station_id = station_id
        self.__issue_time_utc = issue_time_utc
        self.__parameter_series = parameter_series
        self.issue_time = MosmixS.__utc_to_local(self.__issue_time_utc)
        self.date_from = MosmixS.__utc_to_local(self.__parameter_series["Rad1h"].date_from_utc)
        self.date_to = MosmixS.__utc_to_local(self.__parameter_series["Rad1h"].date_to_utc)

    def merge(self, old_mosmix):
        if old_mosmix is None:
            return self
        else:
            min_local_datetime = datetime.now() - timedelta(days=5)
            merged = MosmixS(self.station_id,
                             self.__issue_time_utc,
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
                                "parameter_series": { parameter: self.__parameter_series[parameter].utc_series for parameter in self.__parameter_series.keys()}})
            file.write(data)

    @staticmethod
    def load(filename: str = "mosmix.json"):
        if exists(filename):
            with open(filename, "r") as file:
                try:
                    data = json.loads(file.read())
                    station_id = data['station_id']
                    issue_time_utc = datetime.fromisoformat(data['issue_time_utc'])
                    parameter_series = {parameter: ParameterUtcSeries(data['parameter_series'][parameter]) for parameter in data['parameter_series'].keys()}
                    return MosmixS(station_id, issue_time_utc, parameter_series)
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
        mosmix = MosmixS.load(cache_filename)
        if mosmix is not None:
            if mosmix.is_expired():
                if int((time.time() - os.path.getmtime(cache_filename)) / 60) < 10:   # at maximum all 10 min the (large!) mosmix file will be loaded via web
                    return mosmix
            else:
                return mosmix
        new_mosmix = self.__parent_loader.get()
        merged_mosmix = new_mosmix.merge(mosmix)
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


