import logging
from appdirs import site_data_dir
from threading import RLock
from datetime import datetime, timedelta
from typing import Optional
from typing import List
from pvpower.weather_forecast import WeatherStation, WeatherForecast
from pvpower.traindata import LabelledWeatherForecast, TrainSampleLog
from pvpower.estimator import Estimator



def round_datetime(resolution_minutes: int, dt: datetime = None) -> datetime:
    if dt is None:
        dt = datetime.now()
    rounded_minutes = round(dt.minute / resolution_minutes) * resolution_minutes
    return datetime.strptime(dt.strftime("%d.%m.%Y %H") + ":" + '{0:02d}'.format(rounded_minutes), "%d.%m.%Y %H:%M")


class TestReport:

    def __init__(self, validation_samples: List[LabelledWeatherForecast], predictions: List[int]):
        self.validation_samples = validation_samples
        self.predictions = predictions

    def __percent(self, real, predicted):
        if real == 0:
            return 0
        elif predicted == 0:
            return 0
        else:
            return round((predicted * 100 / real) - 100, 2)

    def __diff_all(self) -> List[float]:
        diff_total = []
        for i in range(len(self.validation_samples)):
            if self.validation_samples[i].irradiance == 0:
                continue
            else:
                predicted = self.predictions[i]

            real = self.validation_samples[i].power_watt
            if real == 0:
                if predicted == 0:     # ignore true 0 predictions to void wasting the score
                    diff = 0
                else:
                    diff = 100
            else:
                if real < 10 and abs(predicted - real) < 10:  # ignore diff < 10
                    diff = 0
                else:
                    diff = self.__percent(real, predicted)
            diff_total.append(diff)
        return sorted(diff_total)

    @property
    def score(self) -> float:
        values = sorted(list(self.__diff_all()))
        values = values[2:-2]
        abs_values = [abs(value) for value in values]
        return round(sum(abs_values) / len(abs_values), 2)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        txt = "score:  " + str(self.score) + "\n"
        txt += '{:14s}        {:14s} {:14s} {:14s} {:14s} {:14s}         {:10s} {:10s}           {:10s}\n'.format("time", "irradiance", "sunshine", "cloud_cover", "visibility", "proba.fog", "real", "predicted", "diff[%]")
        for i in range(0, len(self.validation_samples)):
            txt += '{:<14s}        {:<14d} {:<14d} {:<14d} {:<14d}  {:<14d}        {:<10d} {:<10d}           {:<10d}\n'.format(self.validation_samples[i].time.strftime("%d.%b  %H:%S"),
                                                                                                                               self.validation_samples[i].irradiance,
                                                                                                                               self.validation_samples[i].sunshine,
                                                                                                                               self.validation_samples[i].cloud_cover,
                                                                                                                               self.validation_samples[i].visibility,
                                                                                                                               self.validation_samples[i].probability_for_fog,
                                                                                                                               self.validation_samples[i].power_watt,
                                                                                                                               self.predictions[i],
                                                                                                                               int(self.__percent(self.validation_samples[i].power_watt, self.predictions[i])))
        return txt


class Tester:

    def __init__(self, samples: List[LabelledWeatherForecast]):
        self.__samples = samples

    def evaluate(self, estimator: Estimator) -> List[TestReport]:
        test_reports = []
        samples = estimator.clean_data(self.__samples)
        logging.info("testing with " + str(len(samples)) + " samples")
        for i in range(0, len(samples)):
            # split test data
            num_train_samples = int(len(samples) * 0.7)
            train_samples = samples[0: num_train_samples]
            validation_samples = samples[num_train_samples:]

            # train and test
            estimator.retrain(train_samples)
            predicted = [estimator.predict(test_sample) for test_sample in validation_samples]
            test_reports.append(TestReport(validation_samples, predicted))

            samples = samples[-1:] + samples[:-1]

        test_reports.sort(key=lambda report: report.score)
        return test_reports


class ValueRecorder:

    def __init__(self, resolution_minutes: int = 10):
        self.__resolution_minutes = resolution_minutes
        self.__start_time = round_datetime(resolution_minutes=self.__resolution_minutes)
        self.__end_time = self.__start_time + timedelta(minutes=self.__resolution_minutes)
        self.time = self.__start_time + timedelta(minutes=round(self.__resolution_minutes/2))
        self.__power_values = []

    def is_expired(self):
        return datetime.now() > self.__end_time

    def add(self, value: int):
        self.__power_values.append(value)

    @property
    def average(self) -> Optional[int]:
        if len(self.__power_values) == 0:
            return None
        else:
            return int(sum(self.__power_values) / len(self.__power_values))

    def __str__(self):
        return self.__start_time.strftime("%d.%m.%Y %H:%M") + " -> " + self.__end_time.strftime("%d.%m.%Y %H:%M") + "  average power: " + str(self.average) + " num probes: " + str(len(self.__power_values))


class PvPowerForecast:

    def __init__(self, station_id: str, train_dir: str = None):
        self.__lock = RLock()
        if train_dir is None:
            train_dir = site_data_dir("pv_forecast", appauthor=False)
        self.weather_forecast_service = WeatherStation(station_id)
        self.train_log = TrainSampleLog(train_dir)
        self.__train_value_recorder = ValueRecorder()
        self.__estimator = Estimator()
        self.__date_last_retrain = datetime.now() - timedelta(days=90)
        self.__num_samples_last_retrain = 0
        self.__date_last_weather_forecast = datetime.now() - timedelta(days=1)
        self.__retrain("on initializing")

    @property
    def __max_retrain_period_minutes(self) -> int:
        if self.__num_samples_last_retrain < 1000:
            return 60       # 1 hour
        else:
            return 7*24*60  # 1 week

    def __retrain(self, reason: str):
        try:
            if datetime.now() > (self.__date_last_retrain + timedelta(minutes=self.__max_retrain_period_minutes)):
                samples = self.train_log.all()
                self.__estimator.retrain(samples)
                self.__date_last_retrain = datetime.now()
                self.__num_samples_last_retrain = len(samples)
                if len(samples) < 1000:
                    test_reports = Tester(samples).evaluate(self.__estimator)
                    logging.info("prediction model retrained " + reason + " (median deviation: " + str(round(test_reports[int(len(test_reports)*0.5)].score, 1)) + "% -> smaller is better)")
                else:
                    logging.info("prediction model retrained " + reason)

        except Exception as e:
            logging.warning("error occurred retrain prediction model " + str(e))

    def current_power_reading(self, real_power: int):
        with self.__lock:
            if self.__train_value_recorder.is_expired():
                logging.info(str(self.__train_value_recorder) + " is expired. generated train data record")
                try:
                    if self.__train_value_recorder.average is not None:
                        weather_sample = self.weather_forecast_service.forecast(self.__train_value_recorder.time)
                        annotated_sample = LabelledWeatherForecast(self.__train_value_recorder.time,
                                                                   weather_sample.irradiance,
                                                                   weather_sample.sunshine,
                                                                   weather_sample.cloud_cover,
                                                                   weather_sample.probability_for_fog,
                                                                   weather_sample.visibility,
                                                                   self.__train_value_recorder.average)
                        self.train_log.append(annotated_sample)
                finally:
                    self.__train_value_recorder = ValueRecorder()
                    logging.info(" new value recorder " + str(self.__train_value_recorder))
                    self.__retrain("on updated train data")
            self.__train_value_recorder.add(real_power)

    def predict_by_weather_forecast(self, sample: WeatherForecast) -> int:
        return self.__estimator.predict(sample)

    def predict(self, time: datetime) -> Optional[int]:
        sample = self.weather_forecast_service.forecast(time)
        if sample is None:
            return None
        else:
            return self.predict_by_weather_forecast(sample)