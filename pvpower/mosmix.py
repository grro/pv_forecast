import httpx
import logging
from stream_unzip import stream_unzip
from typing import Dict
from datetime import datetime, timezone
from typing import List
import xml.etree.ElementTree as ET



class IssueTimeCollector:

    def __init__(self):
        self.issue_time = None

    def consume(self, event, elem):
        if elem.tag.endswith("IssueTime"):
            if elem.text is not None:
                time = datetime.fromisoformat(elem.text.replace("Z", "+00:00"))
                local_time = time.replace(tzinfo=timezone.utc).astimezone(tz=None).replace(tzinfo=None)
                self.issue_time = local_time


class TimeStepsCollector:

    def __init__(self):
        self.__is_collecting = False
        self.timesteps = []

    def consume(self, event, elem):
        if event == 'start' and elem.tag.endswith("ForecastTimeSteps"):
            self.timesteps.clear()
            self.__is_collecting = True
        elif event == 'start' and self.__is_collecting and elem.tag.endswith("TimeStep"):
            if elem.text is not None:
                time = datetime.fromisoformat(elem.text.replace("Z", "+00:00"))
                local_time = time.replace(tzinfo=timezone.utc).astimezone(tz=None).replace(tzinfo=None)
                self.timesteps.append(local_time)
        elif self.__is_collecting and event == 'end' and elem.tag.endswith("ForecastTimeSteps"):
            self.__is_collecting = False


class ForecastValuesCollector:

    def __init__(self, station_id: str):
        self.station_id = station_id
        self.__parameter_name = ""
        self.parameters = {}
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

    def __init__(self, name: str, time_series: List[datetime], values: Dict[str, List[float]]):
        self.name = name
        self.series = { time_series[i].strftime("%d.%m.%Y %H"): values[name][i] for i in range(0, len(time_series)) }


class MosmixS:

    # x-check https://mosmix.de/online.html#/station/10724/station
    # parameters (https://dwd-geoportal.de/products/G_FJM/)

    def __init__(self,station_id: str):
        self.station_id = station_id
        self.__fetch_datetime = datetime.now()
        self.__issue_time_collector = IssueTimeCollector()
        self.__timesteps_collector = TimeStepsCollector()
        self.__forecasts_collector = ForecastValuesCollector(station_id)
        self.__rad1h_series = None
        self.__sund1_series = None
        self.__neff_series = None
        self.__wwm_series = None
        self.__vv_series = None


    def consume(self, event, elem):
        self.__issue_time_collector.consume(event, elem)
        self.__timesteps_collector.consume(event, elem)
        self.__forecasts_collector.consume(event, elem)

    def data_from(self) -> datetime:
        return self.__timesteps_collector.timesteps[0]

    def data_to(self) -> datetime:
        return self.__timesteps_collector.timesteps[-1]

    def supports(self, dt: datetime) -> bool:
        return self.data_from().strftime("%Y.%d.%m %H") <= dt.strftime("%Y.%d.%m %H") <= self.data_to().strftime("%Y.%d.%m %H")

    def issue_time(self) -> datetime:
        return self.__issue_time_collector.issue_time

    def content_age_sec(self) -> int:
        return int((datetime.now() - self.issue_time()).total_seconds())

    def elapsed_sec_fetched(self):
        return int((datetime.now() - self.__fetch_datetime).total_seconds())

    def rad1h(self, dt: datetime) -> float:
        if self.__rad1h_series is None:
            self.__rad1h_series = ParameterSeries("Rad1h", self.__timesteps_collector.timesteps, self.__forecasts_collector.parameters).series
        return self.__rad1h_series.get(dt.strftime("%d.%m.%Y %H"))

    def sund1(self, dt: datetime) -> float:
        if self.__sund1_series is None:
            self.__sund1_series = ParameterSeries("SunD1", self.__timesteps_collector.timesteps, self.__forecasts_collector.parameters).series
        return self.__sund1_series.get(dt.strftime("%d.%m.%Y %H"))

    def neff(self, dt: datetime) -> float:
        if self.__neff_series is None:
            self.__neff_series = ParameterSeries("Neff", self.__timesteps_collector.timesteps, self.__forecasts_collector.parameters).series
        return self.__neff_series.get(dt.strftime("%d.%m.%Y %H"))

    def wwm(self, dt: datetime) -> float:
        if self.__wwm_series is None:
            self.__wwm_series = ParameterSeries("wwM", self.__timesteps_collector.timesteps, self.__forecasts_collector.parameters).series
        return self.__wwm_series.get(dt.strftime("%d.%m.%Y %H"))

    def vv(self, dt: datetime) -> float:
        if self.__vv_series is None:
            self.__vv_series = ParameterSeries("VV", self.__timesteps_collector.timesteps, self.__forecasts_collector.parameters).series
        return self.__vv_series.get(dt.strftime("%d.%m.%Y %H"))

    def start_date(self) -> datetime:
        return self.__timesteps_collector.timesteps[0]

    def end_date(self) -> datetime:
        return self.__timesteps_collector.timesteps[-1]

    def __str__(self):
        return self.start_date().strftime("%d.%m.%Y %H:%M") + " -> " + self.end_date().strftime("%d.%m.%Y %H:%M")

    @staticmethod
    def __perform_get_chunked(url):
        with httpx.stream('GET', url) as r:
            yield from r.iter_bytes(chunk_size=65536)

    @staticmethod
    def load(station_id: str, url: str = 'https://opendata.dwd.de/weather/local_forecasts/mos/MOSMIX_S/all_stations/kml/MOSMIX_S_LATEST_240.kmz'):
        logging.debug("loading MOSMIX_S")
        mosmix = MosmixS(station_id)
        xml_parser = ET.XMLPullParser(['start', 'end'])
        for file_name, file_size, unzipped_chunks in stream_unzip(MosmixS.__perform_get_chunked(url)):
            for chunk in unzipped_chunks:
                xml_parser.feed(chunk)
                for event, elem in xml_parser.read_events():
                    mosmix.consume(event, elem)
        logging.info("MOSMIX_S loaded (" + str(mosmix) + ")")
        return mosmix

