import logging
import os
from pathlib import Path
from os.path import exists
from threading import RLock
from datetime import datetime
from typing import List
from dataclasses import dataclass
from pvpower.weather_forecast import WeatherForecast



@dataclass(frozen=True)
class LabelledWeatherForecast(WeatherForecast):
    power_watt: int

    @staticmethod
    def create(weather_forecast: WeatherForecast, power_watt: int):
        return LabelledWeatherForecast(weather_forecast.time,
                                       weather_forecast.irradiance,
                                       weather_forecast.sunshine,
                                       weather_forecast.cloud_cover,
                                       weather_forecast.probability_for_fog,
                                       weather_forecast.visibility,
                                       power_watt)

    @staticmethod
    def __to_string(value):
        return "" if value is None else str(value)

    def to_csv(self) -> str:
        return self.time.strftime("%d.%m.%Y %H:%M") + ";" + \
               LabelledWeatherForecast.__to_string(self.power_watt) + ";" + \
               LabelledWeatherForecast.__to_string(self.irradiance) + ";" + \
               LabelledWeatherForecast.__to_string(self.sunshine) + ";" + \
               LabelledWeatherForecast.__to_string(self.cloud_cover) + ";" + \
               LabelledWeatherForecast.__to_string(self.probability_for_fog) + ";" + \
               LabelledWeatherForecast.__to_string(self.visibility)

    @staticmethod
    def csv_header() -> str:
        return "time;real_pv_power;irradiance;sunshine;cloud_cover;probability_for_fog;visibility"

    @staticmethod
    def __to_int(txt):
        if len(txt) > 0:
            return int(float(txt))
        else:
            return None

    @staticmethod
    def from_csv(line: str):
        parts = line.split(";")
        time = datetime.strptime(parts[0], "%d.%m.%Y %H:%M")
        real_pv_power = LabelledWeatherForecast.__to_int(parts[1])
        irradiance = LabelledWeatherForecast.__to_int(parts[2])
        sunshine = LabelledWeatherForecast.__to_int(parts[3])
        cloud_cover_effective = LabelledWeatherForecast.__to_int(parts[4])
        probability_for_fog = LabelledWeatherForecast.__to_int(parts[5])
        visibility = LabelledWeatherForecast.__to_int(parts[6])
        sample = LabelledWeatherForecast(time, irradiance, sunshine, cloud_cover_effective, probability_for_fog, visibility, real_pv_power)
        return sample


class TrainSampleLog:

    def __init__(self, dirname: str):
        self.lock = RLock()
        self.__dirname = dirname
        logging.info("using train file " + self.filename)

    @property
    def filename(self):
        fn = os.path.join(self.__dirname, "train.csv")
        if not exists(fn):
            directory = Path(fn).parent
            if not exists(directory):
                os.makedirs(directory)
        return fn

    def append(self, sample: LabelledWeatherForecast):
        with self.lock:
            exits = exists(self.filename)
            with open(self.filename, "ab") as file:
                if not exits:
                    file.write((LabelledWeatherForecast.csv_header() + "\n").encode(encoding='UTF-8'))
                line = sample.to_csv() + "\n"
                file.write(line.encode(encoding='UTF-8'))
                logging.info("record appended to train file " + self.filename)

    def all(self) -> List[LabelledWeatherForecast]:
        with self.lock:
            if exists(self.filename):
                try:
                    with open(self.filename, "rb") as file:
                        lines = [raw_line.decode('UTF-8').strip() for raw_line in file.readlines()]
                        samples = []
                        for line in lines:
                            try:
                                samples.append(LabelledWeatherForecast.from_csv(line))
                            except Exception as e:
                                pass
                        return samples
                except Exception as e:
                    logging.warning("error occurred loading " + self.filename + " " + str(e))
        return []

    def __str__(self):
        return "\n".join([sample.to_csv() for sample in self.all()])



