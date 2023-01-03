import logging
import os
import pytz
import shutil
import gzip
import tempfile
from time import sleep
from appdirs import site_data_dir
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
                 cloud_cover_effective: int,
                 probability_for_fog: int,
                 visibility: int,
                 power_watt: int):
        super().__init__(time, irradiance, sunshine, cloud_cover_effective, probability_for_fog, visibility)
        self.power_watt = power_watt


    @staticmethod
    def create(weather_forecast: WeatherForecast, power_watt: int, time: datetime = None):

        return LabelledWeatherForecast(weather_forecast.time_utc if time is None else time,
                                       weather_forecast.irradiance,
                                       weather_forecast.sunshine,
                                       weather_forecast.cloud_cover_effective,
                                       weather_forecast.probability_for_fog,
                                       weather_forecast.visibility,
                                       power_watt)

    @staticmethod
    def __to_string(value):
        return "" if value is None else str(value)

    @staticmethod
    def csv_header() -> str:
        return "utc_time;real_pv_power;irradiance;sunshine;cloud_cover;probability_for_fog;visibility"

    @staticmethod
    def __to_int(txt) -> Optional[int]:
        if len(txt) > 0:
            return int(float(txt))
        else:
            return None

    def to_csv(self) -> str:
        utc_time = self.time_utc
        return utc_time.isoformat() + ";" + \
               LabelledWeatherForecast.__to_string(self.power_watt) + ";" + \
               LabelledWeatherForecast.__to_string(self.irradiance) + ";" + \
               LabelledWeatherForecast.__to_string(self.sunshine) + ";" + \
               LabelledWeatherForecast.__to_string(self.cloud_cover_effective) + ";" + \
               LabelledWeatherForecast.__to_string(self.probability_for_fog) + ";" + \
               LabelledWeatherForecast.__to_string(self.visibility)

    @staticmethod
    def from_csv(line: str):
        parts = line.split(";")
        utc_time = datetime.fromisoformat(parts[0])
        real_pv_power = LabelledWeatherForecast.__to_int(parts[1])
        irradiance = LabelledWeatherForecast.__to_int(parts[2])
        sunshine = LabelledWeatherForecast.__to_int(parts[3])
        cloud_cover_effective = LabelledWeatherForecast.__to_int(parts[4])
        probability_for_fog = LabelledWeatherForecast.__to_int(parts[5])
        visibility = LabelledWeatherForecast.__to_int(parts[6])
        sample = LabelledWeatherForecast(utc_time, irradiance, sunshine, cloud_cover_effective, probability_for_fog, visibility, real_pv_power)
        return sample


    def __str__(self):
        return super().__str__() + ", power_watt=" + str(self.power_watt)


class TrainData:

    def __init__(self, samples: List[LabelledWeatherForecast]):
        self.samples = self.__clean_data(samples)

    def __iter__(self):
        return iter(self.samples)

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def __clean_data(samples: List[LabelledWeatherForecast]) -> List[LabelledWeatherForecast]:
        seen = list()
        return list(filter(lambda sample: seen.append(sample.time_utc) is None if sample.time_utc not in seen else False, samples))  # remove duplicates

    def rotated(self, offset_percent: int):
        step = int(offset_percent * len(self.samples) / 100)
        return TrainData(self.samples[step:] + self.samples[:step])

    def split(self, ratio_percent:int = 67):
        size = len(self.samples)
        left = []
        right = []
        for i in range(0, size):
            if i > round(size * ratio_percent / 100, 0):
                right.append(self.samples[i])
            else:
                left.append(self.samples[i])
        return TrainData(left), TrainData(right)

    def __str__(self):
        return "train data size " + str(len(self.samples)) + " (" + self.samples[0].time.strftime("%Y.%m.%dT%H:%M") + " ... " + self.samples[-1].time.strftime("%Y.%m.%dT%H:%M") + ")"


class TrainSampleLog:
    COMPACTION_PERIOD_DAYS = 15
    FILENAME = "train.csv.gz"

    def __init__(self, dirname: str = None):
        self.lock = RLock()
        self.__dirname = dirname if dirname is not None else site_data_dir("pv_power", appauthor=False)
        self.__last_compaction_time = datetime.now() - timedelta(days=self.COMPACTION_PERIOD_DAYS*2)

    def filename(self):
        fn = os.path.join(self.__dirname, self.FILENAME)
        if not exists(fn):
            directory = Path(fn).parent
            if not exists(directory):
                os.makedirs(directory)
        return fn

    def append(self, sample: LabelledWeatherForecast):
        with self.lock:
            compr_fn = self.filename()
            with gzip.open(compr_fn, "ab") as file:
                line = sample.to_csv() + "\n"
                file.write(line.encode(encoding='UTF-8'))

        if datetime.now() > (self.__last_compaction_time + timedelta(days=self.COMPACTION_PERIOD_DAYS)):
            self.__last_compaction_time = datetime.now()
            Thread(target=self.compact, args=(15,), daemon=True).start()

    def all(self) -> TrainData:
        with self.lock:
            compr_fn = self.filename()
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
                        return TrainData(samples)
                except Exception as e:
                    logging.warning("error occurred loading " + compr_fn + " " + str(e))
        return TrainData([])

    def compact(self, delay_sec:int = 0):
        sleep(delay_sec)

        fn = self.filename()
        train_data = self.all()
        min_datetime = datetime.now() - timedelta(days=4*365)

        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_file = os.path.join(tmp_dir, 'traindata.csv')
            with gzip.open(temp_file, "wb") as file:
                file.write((LabelledWeatherForecast.csv_header() + "\n").encode(encoding='UTF-8'))
                num_written = 0
                for sample in train_data:
                    if not sample.time_utc < min_datetime.astimezone(pytz.UTC):
                        line = sample.to_csv() + "\n"
                        file.write(line.encode(encoding='UTF-8'))
                        num_written += 1
            shutil.move(temp_file, fn)
            logging.info("train file " + fn + " compacted  (" + str(len(train_data)) + " > " + str(num_written) + ")")

    def __str__(self):
        return "\n".join([sample.to_csv() for sample in self.all()])



