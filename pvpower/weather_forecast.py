import logging
import requests
import xmltodict
from io import BytesIO
from zipfile import ZipFile
from typing import Optional, Any
from dataclasses import dataclass
from typing import Dict
from datetime import datetime, timedelta
from typing import List



@dataclass(frozen=True)
class WeatherForecast:
    time: datetime
    irradiance: int
    sunshine: int
    cloud_cover: int
    probability_for_fog: int
    visibility: int


class WeatherStation:

    def __init__(self, station: str, refresh_period_min: int = 9):
        self.url = 'https://opendata.dwd.de/weather/local_forecasts/mos/MOSMIX_L/single_stations/' + station + '/kml/MOSMIX_L_LATEST_K887.kmz'
        self.__refresh_period_min = refresh_period_min
        self.__last_refresh = datetime.now() - timedelta(days=2)
        self.__global_irradiance = {}
        self.__sunshine = {}
        self.__cloud_cover_effective = {}
        self.__probability_for_fog = {}
        self.__visibility = {}

    @staticmethod
    def __datetime_to_string(date: datetime) -> str:
        return date.strftime("%d.%m.%Y %H")

    def __read_values_by_datetime(self, forecast: List[Dict[str, Any]], time_steps: List[datetime], field_name: str):
        values = [param['dwd:value'].split() for param in forecast if param['@dwd:elementName'] == field_name][0]
        return {WeatherStation.__datetime_to_string(time_steps[i]): int(float(values[i])) for i in range(0, len(time_steps))}

    def __refresh_data(self, force: bool = False):
        if force or datetime.now() > self.__last_refresh + timedelta(minutes=self.__refresh_period_min):
            response = requests.get(self.url, timeout=60)
            if 300 > response.status_code >= 200:
                with ZipFile(BytesIO(response.content)) as my_zip_file:
                    for contained_file in my_zip_file.namelist():
                        doc = xmltodict.parse(my_zip_file.open(contained_file).read())

                        # read time steps
                        time_steps_iso = doc['kml:kml']['kml:Document']['kml:ExtendedData']['dwd:ProductDefinition']['dwd:ForecastTimeSteps']['dwd:TimeStep']
                        time_steps = [datetime.strptime(time_step, '%Y-%m-%dT%H:%M:%S.%fZ') for time_step in time_steps_iso]

                        # read parameters (https://dwd-geoportal.de/products/G_FJM/)
                        forecast = doc['kml:kml']['kml:Document']['kml:Placemark']['kml:ExtendedData']['dwd:Forecast']
                        self.__global_irradiance = self.__read_values_by_datetime(forecast, time_steps, 'Rad1h')
                        self.__sunshine = self.__read_values_by_datetime(forecast, time_steps, 'SunD1')
                        self.__cloud_cover_effective = self.__read_values_by_datetime(forecast, time_steps, 'Neff')
                        self.__probability_for_fog = self.__read_values_by_datetime(forecast, time_steps, 'wwM')
                        self.__visibility = self.__read_values_by_datetime(forecast, time_steps, 'VV')

                    self.__last_refresh = datetime.now()
            else:
                logging.warning("error occurred calling " + self.url + " Got " + str(response.status_code) + " " + str(response.text))

    def __forecast_value(self, date: datetime, forecast_list: Dict[str,int]) -> Optional[int]:
        date_string = WeatherStation.__datetime_to_string(date)
        if date_string not in forecast_list.keys():
            self.__refresh_data(True)
        return forecast_list.get(date_string, None)

    def forecast(self, time: datetime = None) -> WeatherForecast:
        time = time if time is not None else datetime.now()
        time = datetime.strptime(time.strftime("%d.%m.%Y %H"), "%d.%m.%Y %H")
        self.__refresh_data(False)
        irradiance = self.__forecast_value(time, self.__global_irradiance)
        sunshine = self.__forecast_value(time, self.__sunshine)
        cloud_cover_effective = self.__forecast_value(time, self.__cloud_cover_effective)
        probability_for_fog = self.__forecast_value(time, self.__probability_for_fog)
        visibility = self.__forecast_value(time, self.__visibility)
        return WeatherForecast(time, irradiance, sunshine, cloud_cover_effective, probability_for_fog, visibility)
