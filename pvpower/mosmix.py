import os.path

import httpx
import json
from stream_unzip import stream_unzip
from typing import Dict
import pytz
from appdirs import site_data_dir
from os.path import exists
from datetime import datetime, timezone
from typing import List
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


class ParameterSeries:

    def __init__(self, name: str, utc_time_series: List[datetime], values: Dict[str, List[float]]):
        self.name = name
        self.series = { utc_time_series[i].strftime("%d.%m.%Y %H"): values[name][i] for i in range(0, len(utc_time_series)) }




class MosmixS:

    def __init__(self,
                 station_id: str,
                 issue_time_utc: datetime,
                 utc_timesteps: List[datetime],
                 parameters: Dict[str, List[float]]):
        self.station_id = station_id
        self.issue_time_utc = issue_time_utc
        self.__utc_timesteps = utc_timesteps
        self.__parameters = parameters
        self.__parameter_series = {}

    def is_expired(self) -> bool:
        content_age_sec = int((datetime.now(timezone.utc) - self.issue_time_utc).total_seconds())
        return content_age_sec > (90*60)

    def utc_date_from(self) -> datetime:
        return self.__utc_timesteps[0]

    def utc_date_to(self) -> datetime:
        return self.__utc_timesteps[-1]

    def supports(self, dt: datetime) -> bool:
        return self.utc_date_from() <= dt.astimezone(pytz.UTC) <= self.utc_date_to()

    def __read(self, parameter: str, dt: datetime) -> float:
        if parameter not in self.__parameter_series.keys():
            self.__parameter_series[parameter] = ParameterSeries(parameter, self.__utc_timesteps, self.__parameters).series
        utc_time = dt.astimezone(pytz.UTC)
        value = self.__parameter_series.get(parameter).get(utc_time.strftime("%d.%m.%Y %H"))
        #logging.debug("got " + str(value) + " for requested time " + dt.isoformat() + " (utc: " + utc_time.isoformat() + ")")
        return value

    def rad1h(self, dt: datetime) -> float:
        return self.__read("Rad1h", dt)

    def sund1(self, dt: datetime) -> float:
        return self.__read("SunD1", dt)

    def neff(self, dt: datetime) -> float:
        return self.__read("Neff", dt)

    def wwm(self, dt: datetime) -> float:
        return self.__read("wwM", dt)

    def vv(self, dt: datetime) -> float:
        return self.__read("VV", dt)

    def __str__(self):
        return self.utc_date_from().strftime("%d.%m.%Y %H:%M") + " utc -> " + self.utc_date_to().strftime("%d.%m.%Y %H:%M") + " utc"

    def save(self, filename: str = "mosmix.json"):
        with open(filename, "w") as file:
            data = json.dumps({ "station_id": self.station_id,
                                "issue_time_utc": self.issue_time_utc.isoformat(),
                                "utc_timesteps": [time.isoformat() for time in self.__utc_timesteps],
                                "parameters": self.__parameters
                                })
            file.write(data)

    @staticmethod
    def load(filename: str = "mosmix.json"):
        if exists(filename):
            with open(filename, "r") as file:
                data = json.loads(file.read())
                station_id = data['station_id']
                issue_time_utc = datetime.fromisoformat(data['issue_time_utc'])
                utc_timesteps = [datetime.fromisoformat(time) for time in data['utc_timesteps']]
                parameters = data['parameters']
                return MosmixS(station_id,
                               issue_time_utc,
                               utc_timesteps,
                               parameters)
        else:
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
    def load(station_id: str):
        cache_filename = os.path.join(site_data_dir("pv_forecast", appauthor=False), "mosmixs_" + station_id + ".json")
        mosmix = MosmixS.load(cache_filename)
        if mosmix is not None and not mosmix.is_expired():
            return mosmix

        url = 'https://opendata.dwd.de/weather/local_forecasts/mos/MOSMIX_S/all_stations/kml/MOSMIX_S_LATEST_240.kmz'
        mosmix_loader = MosmixSWeb(station_id)
        xml_parser = ET.XMLPullParser(['start', 'end'])
        for file_name, file_size, unzipped_chunks in stream_unzip(MosmixSWeb.__perform_get_chunked(url)):
            for chunk in unzipped_chunks:
                xml_parser.feed(chunk)
                for event, elem in xml_parser.read_events():
                    mosmix_loader.consume(event, elem)
        mosmix = MosmixS(station_id,
                         mosmix_loader.__issue_time_collector.issue_time_utc,
                         mosmix_loader.__timesteps_collector.utc_timesteps,
                         mosmix_loader.__forecasts_collector.parameters)

        if cache_filename is not None:
            mosmix.save(cache_filename)
        return mosmix


