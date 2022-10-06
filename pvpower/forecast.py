import logging
import os
from pathlib import Path
from os.path import exists
from appdirs import site_data_dir
from threading import RLock
from datetime import datetime, timedelta
from typing import Optional
from sklearn import svm
from typing import List
from dataclasses import dataclass
from pvpower.weather import WeatherStation, WeatherForecast


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

    @property
    def filename(self):
        fn = os.path.join(self.__dirname, "train.csv")
        if not exists(fn):
            dir = Path(fn).parent
            if not exists(dir):
                os.makedirs(dir)
        return fn

    def append(self, sample: LabelledWeatherForecast):
        with self.lock:
            exits = exists(self.filename)
            #with gzip.GzipFile(self.__filename + ".json.gz", "ab") as file:
            with open(self.filename, "ab") as file:   #remove me
                if not exits:
                    file.write((LabelledWeatherForecast.csv_header() + "\n").encode(encoding='UTF-8'))
                line = sample.to_csv() + "\n"
                file.write(line.encode(encoding='UTF-8'))

    def all(self) -> List[LabelledWeatherForecast]:
        with self.lock:
            #if exists(self.__filename + ".json.gz"):
            if exists(self.filename):
                try:
                    #with gzip.GzipFile(self.__filename + ".json.gz", "rb") as file:
                    with open(self.filename, "rb") as file:   # remove me
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



class Estimator:

    def __init__(self):
        # it seems that the SVM approach produces good predictions
        # refer https://www.sciencedirect.com/science/article/pii/S136403212200274X?via%3Dihub and https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.221.4021&rep=rep1&type=pdf
        kernel = 'poly'
        self.clf = svm.SVC(kernel='rbf')
        self.num_samples_last_train = 0
        logging.info("using SVM kernel=" + kernel + " vectorizer=month,hour,irradiance")

    def __scale(self, value: int, max_value: int, digits=0) -> float:
        if value == 0:
            return 0
        else:
            return round(value * 100 / max_value, digits)

    def __vectorize(self, sample: WeatherForecast) -> List[float]:
        return [self.__scale(sample.time.month, 12),
                self.__scale(sample.time.hour, 24),
                #self.__scale(sample.sunshine, 10000),
                #self.__scale(sample.probability_for_fog, 100),
                #self.__scale(sample.visibility, 100000),
                self.__scale(sample.irradiance, 1000)]

    def retrain(self, samples: List[LabelledWeatherForecast]):
        samples = [sample for sample in samples if sample.irradiance > 0]
        num_samples = len(samples)
        if self.num_samples_last_train != num_samples:
            if num_samples < 2:
                logging.warning("just " + str(len(samples)) + " samples with irradiance > 0 are available. At least 2 samples are required")
            else:
                feature_vector_list = [self.__vectorize(sample) for sample in samples]
                label_list = [sample.power_watt for sample in samples]
                if len(set(label_list)) > 1:
                    logging.info("retrain prediction model with " + str(num_samples) + " samples")
                    self.clf.fit(feature_vector_list, label_list)
                    self.num_samples_last_train = num_samples
                else:
                    logging.info("ignore retrain. Retrain requires more than " + str(len(set(label_list))) + " classes (samples " + str(num_samples) + ")")

    def predict(self, sample: WeatherForecast) -> Optional[int]:
        try:
            if sample.irradiance > 0:
                feature_vector = self.__vectorize(sample)
                #print(feature_vector)
                predicted = self.clf.predict([feature_vector])[0]
                return int(predicted)
            else:
                return 0
        except Exception as e:
            logging.warning("error occurred predicting " + str(sample) + " " + str(e))
            return None


class ValueRecorder:

    def __init__(self):
        self.__time = datetime.now()
        self.__power_values_of_current_hour = []

    def same_hour(self, other_time: datetime):
        return self.__time.strftime("%d.%m.%Y %H") == other_time.strftime("%d.%m.%Y %H")

    @property
    def time(self) -> datetime:
        return datetime.strptime(self.__time.strftime("%d.%m.%Y %H"), "%d.%m.%Y %H")

    @property
    def average(self) -> Optional[int]:
        if len(self.__power_values_of_current_hour) == 0:
            return None
        else:
            return int(sum(self.__power_values_of_current_hour) / len(self.__power_values_of_current_hour))

    def add(self, value: int):
        self.__power_values_of_current_hour.append(value)

    def reset(self):
        self.__time = datetime.now()
        self.__power_values_of_current_hour = []



class PvPowerForecast:

    def __init__(self, station_id: str, train_dir: str = None):
        if train_dir is None:
            train_dir = site_data_dir("pv_forecast", appauthor=False)
        self.weather_forecast_service = WeatherStation(station_id)
        self.train_log = TrainSampleLog(train_dir)
        self.__train_data_value_recorder = ValueRecorder()
        self.__estimator = Estimator()
        self.__date_last_retrain = datetime.now() - timedelta(days=90)
        self.__num_samples_last_retrain = 0
        self.__date_last_weather_forecast = datetime.now() - timedelta(days=1)
        logging.info("using " + train_dir + " dir to store train data")

    def __retrain(self):
        try:
            minutes_since_last_retrain = int((datetime.now() - self.__date_last_retrain).total_seconds() / 60)
            min_minutes = 60 if (self.__num_samples_last_retrain < (30 * 24)) else (7*24*60)  # with first month each hour else each week
            if minutes_since_last_retrain > min_minutes:   # at maximum 1 train per day
                train_data = self.train_log.all()
                self.__estimator.retrain(train_data)
                self.__date_last_retrain = datetime.now()
                self.__num_samples_last_retrain = len(train_data)
        except Exception as e:
            logging.warning("error occurred retrain prediction model " + str(e))

    def current_power_reading(self, real_power: int):
        if self.__train_data_value_recorder.same_hour(datetime.now()):
            self.__train_data_value_recorder.add(real_power)
        else:
            if self.__train_data_value_recorder.average is not None:
                weather_sample = self.weather_forecast_service.forecast(self.__train_data_value_recorder.time)
                annotated_sample = LabelledWeatherForecast(self.__train_data_value_recorder.time,
                                                           weather_sample.irradiance,
                                                           weather_sample.sunshine,
                                                           weather_sample.cloud_cover,
                                                           weather_sample.probability_for_fog,
                                                           weather_sample.visibility,
                                                           self.__train_data_value_recorder.average)
                self.train_log.append(annotated_sample)
                self.__retrain()
            self.__train_data_value_recorder.reset()

    def predict_by_weather_forecast(self, sample: WeatherForecast) -> int:
        self.__retrain()
        return self.__estimator.predict(sample)

    def predict(self, time: datetime) -> Optional[int]:
        sample = self.weather_forecast_service.forecast(time)
        if sample is None:
            return None
        else:
            return self.predict_by_weather_forecast(sample)