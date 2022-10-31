import logging
import os.path
import httpx
import json
from stream_unzip import stream_unzip
from random import randrange
import pytz
from appdirs import site_data_dir
from os.path import exists
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any
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
        self.__series = series

    def size(self) -> int:
        return len(self.__series.keys())

    def value_at(self, dt: datetime) -> float:
        return self.__series.get(dt.astimezone(pytz.UTC).strftime("%Y.%m.%d %H"))

    def to_dict(self) -> Dict[str, Any]:
        return { "name": self.name,
                 "series": self.__series }

    def merge(self, other, min_datetime: datetime):
        yesterday = min_datetime.astimezone(pytz.UTC).strftime("%Y.%m.%d %H")
        merged_series = {}
        for time in other.__series.keys():
            if time >= yesterday:
                merged_series[time] = other.__series[time]
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
               utc_timesteps: List[datetime],
               parameters: Dict[str, List[float]]):
        return MosmixS(station_id,
                       issue_time_utc,
                       utc_timesteps[0],
                       utc_timesteps[-1],
                       {parameter: ParameterUtcSeries.create(parameter, utc_timesteps, parameters) for parameter in parameters})

    def __init__(self,
                 station_id: str,
                 issue_time_utc: datetime,
                 utc_date_from: datetime,
                 utc_date_to: datetime,
                 parameter_series: Dict[str, ParameterUtcSeries]):
        self.station_id = station_id
        self.issue_time_utc = issue_time_utc
        self.utc_date_from = utc_date_from
        self.utc_date_to = utc_date_to
        self.__parameter_series = parameter_series

    def merge(self, old_mosmix, min_datetime: datetime):
        if old_mosmix is None:
            return self
        else:
            merged = MosmixS(self.station_id,
                             self.issue_time_utc,
                             old_mosmix.utc_date_from,
                             self.utc_date_to,
                             {parameter: self.__parameter_series[parameter].merge(old_mosmix.__parameter_series[parameter], min_datetime) for parameter in self.__parameter_series.keys()} )
            logging.debug("merging \nold mosmix: " + str(old_mosmix) + " \nnew mosmix: " + str(self) + " \n-> " + str(merged))
            return merged

    def is_expired(self) -> bool:
        content_age_sec = int((datetime.now(timezone.utc) - self.issue_time_utc).total_seconds())
        return content_age_sec > (60*60 + 25*60 + randrange(15)*60)

    def supports(self, dt: datetime) -> bool:
        return self.utc_date_from <= dt.astimezone(pytz.UTC) <= self.utc_date_to

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
        return "issued=" + self.issue_time_utc.strftime("%d.%m.%Y %H:%M") + "/" + str(self.__parameter_series["Rad1h"].size()) + " entries " + self.utc_date_from.strftime("%d.%m.%Y %H:%M") + " utc -> " + self.utc_date_to.strftime("%d.%m.%Y %H:%M") + " utc"

    def save(self, filename: str = "mosmix.json"):
        with open(filename, "w") as file:
            data = json.dumps({ "station_id": self.station_id,
                                "issue_time_utc": self.issue_time_utc.isoformat(),
                                "utc_date_from": self.utc_date_from.isoformat(),
                                "utc_date_to": self.utc_date_to.isoformat(),
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


class MosmixSWeb:

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
    def __cachedir() -> str:
        dir = site_data_dir("pv_forecast", appauthor=False)
        if not exists(dir):
            os.makedirs(dir)
        return dir

    @staticmethod
    def load(station_id: str):
        # load cached mosmix
        cache_filename = os.path.join(MosmixSWeb.__cachedir(), "mosmixs_" + station_id + ".json")
        cached_mosmix = MosmixS.load(cache_filename)
        if cached_mosmix is not None and not cached_mosmix.is_expired():
            return cached_mosmix

        # cached mosmix is expired
        url = 'https://opendata.dwd.de/weather/local_forecasts/mos/MOSMIX_S/all_stations/kml/MOSMIX_S_LATEST_240.kmz'
        mosmix_loader = MosmixSWeb(station_id)
        xml_parser = ET.XMLPullParser(['start', 'end'])
        for file_name, file_size, unzipped_chunks in stream_unzip(MosmixSWeb.__perform_get_chunked(url)):
            for chunk in unzipped_chunks:
                xml_parser.feed(chunk)
                for event, elem in xml_parser.read_events():
                    mosmix_loader.consume(event, elem)
        mosmix = MosmixS.create(station_id,
                                mosmix_loader.__issue_time_collector.issue_time_utc,
                                mosmix_loader.__timesteps_collector.utc_timesteps,
                                mosmix_loader.__forecasts_collector.parameters)

        mosmix = mosmix.merge(cached_mosmix, datetime.now() - timedelta(days=1))
        mosmix.save(cache_filename)
        return mosmix


