import logging
import os
import pytz
import shutil
import gzip
import tempfile
from time import sleep
from threading import Thread
from pathlib import Path
from os.path import exists
from threading import RLock
from datetime import datetime, timedelta
from typing import List, Optional
from pvpower.weather_forecast import WeatherForecast


class LabelledWeatherForecast(WeatherForecast):

    def __init__(self,
                 time: datetime,
                 irradiance: int,
                 sunshine: int,
                 cloud_cover: int,
                 probability_for_fog: int,
                 visibility: int,
                 power_watt: int):
        super().__init__(time, irradiance, sunshine, cloud_cover, probability_for_fog, visibility)
        self.power_watt = power_watt


    @staticmethod
    def create(weather_forecast: WeatherForecast, power_watt: int, time: datetime = None):

        return LabelledWeatherForecast(weather_forecast.time_utc if time is None else time,
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
        utc_time = self.time_utc
        return utc_time.strftime("%d.%m.%Y %H:%M") + ";" + \
               LabelledWeatherForecast.__to_string(self.power_watt) + ";" + \
               LabelledWeatherForecast.__to_string(self.irradiance) + ";" + \
               LabelledWeatherForecast.__to_string(self.sunshine) + ";" + \
               LabelledWeatherForecast.__to_string(self.cloud_cover) + ";" + \
               LabelledWeatherForecast.__to_string(self.probability_for_fog) + ";" + \
               LabelledWeatherForecast.__to_string(self.visibility)

    @staticmethod
    def csv_header() -> str:
        return "utc_time;real_pv_power;irradiance;sunshine;cloud_cover;probability_for_fog;visibility"

    @staticmethod
    def __to_int(txt) -> Optional[int]:
        if len(txt) > 0:
            return int(float(txt))
        else:
            return None

    @staticmethod
    def from_csv(line: str):
        parts = line.split(";")
        utc_time = datetime.strptime(parts[0] + ":00+00:00", "%d.%m.%Y %H:%M:%S%z")
        real_pv_power = LabelledWeatherForecast.__to_int(parts[1])
        irradiance = LabelledWeatherForecast.__to_int(parts[2])
        sunshine = LabelledWeatherForecast.__to_int(parts[3])
        cloud_cover_effective = LabelledWeatherForecast.__to_int(parts[4])
        probability_for_fog = LabelledWeatherForecast.__to_int(parts[5])
        visibility = LabelledWeatherForecast.__to_int(parts[6])
        sample = LabelledWeatherForecast(utc_time, irradiance, sunshine, cloud_cover_effective, probability_for_fog, visibility, real_pv_power)
        return sample


class TrainSampleLog:
    COMPACTION_PERIOD_DAYS = 10

    def __init__(self, dirname: str):
        self.lock = RLock()
        self.__dirname = dirname
        self.__last_compaction_time = datetime.now() - timedelta(days=self.COMPACTION_PERIOD_DAYS*2)

    def filename(self, compressed: bool = False):
        if compressed:
            fn = os.path.join(self.__dirname, "train.csv.gz")
        else:
            fn = os.path.join(self.__dirname, "train.csv")
        if not exists(fn):
            directory = Path(fn).parent
            if not exists(directory):
                os.makedirs(directory)
        return fn

    def append(self, sample: LabelledWeatherForecast):
        with self.lock:
            # compressed file
            compr_fn = self.filename(compressed=True)
            with gzip.open(compr_fn, "ab") as file:
                line = sample.to_csv() + "\n"
                file.write(line.encode(encoding='UTF-8'))

        if datetime.now() > (self.__last_compaction_time + timedelta(days=self.COMPACTION_PERIOD_DAYS)):
            self.__last_compaction_time = datetime.now()
            Thread(target=self.compact, args=(15,), daemon=True).start()

    def all(self) -> List[LabelledWeatherForecast]:
        with self.lock:
            # plain file
            fn = self.filename()
            if exists(fn):
                try:
                    with open(fn, "rb") as file:
                        lines = [raw_line.decode('UTF-8').strip() for raw_line in file.readlines()]
                        samples = []
                        for line in lines:
                            try:
                                samples.append(LabelledWeatherForecast.from_csv(line))
                            except Exception as e:
                                logging.warning(e)
                        return samples
                except Exception as e:
                    logging.warning("error occurred loading " + fn + " " + str(e))

            # compressed file
            compr_fn = self.filename(compressed=True)
            if exists(compr_fn):
                try:
                    with gzip.open(compr_fn, "rb") as file:
                        lines = [raw_line.decode('UTF-8').strip() for raw_line in file.readlines()]
                        samples = []
                        for line in lines:
                            try:
                                samples.append(LabelledWeatherForecast.from_csv(line))
                            except Exception as e:
                                pass
                        return samples
                except Exception as e:
                    logging.warning("error occurred loading " + compr_fn + " " + str(e))
        return []

    def compact(self, delay_sec:int = 0):
        sleep(delay_sec)

        fn = self.filename(compressed=True)
        samples = self.all()
        min_datetime = datetime.now() - timedelta(days=400)

        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_file = os.path.join(tmp_dir, 'traindata.csv')
            with gzip.open(temp_file, "wb") as file:
                file.write((LabelledWeatherForecast.csv_header() + "\n").encode(encoding='UTF-8'))
                num_samples = len(samples)
                num_survived = 0
                previous_sample = None
                for sample in samples:
                    expired = sample.time_utc < min_datetime.astimezone(pytz.UTC)
                    duplicate = False
                    if previous_sample is not None:
                        duplicate = previous_sample.time_utc == sample.time_utc
                    previous_sample = sample
                    if not expired and not duplicate:
                        num_survived += 1
                        line = sample.to_csv() + "\n"
                        file.write(line.encode(encoding='UTF-8'))
            shutil.move(temp_file, fn)
            logging.info("train file " + fn + " compacted (" + str(num_samples - num_survived) + " of " + str(num_samples) + " removed)")

    def __str__(self):
        return "\n".join([sample.to_csv() for sample in self.all()])



