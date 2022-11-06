import logging
import pytz
from dataclasses import dataclass
from datetime import datetime
from abc import ABC, abstractmethod
from sklearn import svm
from typing import List
from pvpower.weather_forecast import WeatherForecast
from pvpower.traindata import LabelledWeatherForecast



class Vectorizer(ABC):

    def __init__(self, datetime_resolution_minutes: int = 20):
        self._datetime_resolution_minutes = datetime_resolution_minutes

    def _utc_minutes_of_day(self, dt: datetime) -> int:
        utc_time = dt.astimezone(pytz.UTC)
        return (utc_time.hour * 60) + utc_time.minute

    def _scale(self, value: int, max_value: int, digits=1) -> float:
        if value == 0:
            return 0
        else:
            return round(value * 100 / max_value, digits)

    @abstractmethod
    def vectorize(self, sample: WeatherForecast) -> List[float]:
        pass



class CoreVectorizer(Vectorizer):

    def vectorize(self, sample: WeatherForecast) -> List[float]:
        vectorized = [self._scale(sample.time_utc.month, 12),
                      self._scale(int(self._utc_minutes_of_day(sample.time_utc) / 10), int((24 * 60) / 10)),
                      self._scale(sample.irradiance, 1000)]
        return vectorized

    def __str__(self):
        return "CoreVectorizer"


class PlusVisibilityVectorizer(CoreVectorizer):

    def vectorize(self, sample: WeatherForecast) -> List[float]:
        return super().vectorize(sample) +  [self._scale(sample.visibility, 50000)]

    def __str__(self):
        return "Core+VisibilityVectorizer"


class PlusSunshineVectorizer(CoreVectorizer):

    def vectorize(self, sample: WeatherForecast) -> List[float]:
        return super().vectorize(sample) +  [self._scale(sample.sunshine, 5000)]

    def __str__(self):
        return "Core+SunshineVectorizer"


class PlusCloudCoverVectorizer(CoreVectorizer):

    def vectorize(self, sample: WeatherForecast) -> List[float]:
        return super().vectorize(sample) +  [self._scale(sample.cloud_cover, 200)]

    def __str__(self):
        return "Core+CloudVectorizer"


class PlusVisibilitySunshineVectorizer(CoreVectorizer):

    def vectorize(self, sample: WeatherForecast) -> List[float]:
        return super().vectorize(sample) +  [self._scale(sample.visibility, 50000),
                                             self._scale(sample.sunshine, 5000)]

    def __str__(self):
        return "Core+Visibility+SunshineVectorizer"


class PlusVisibilityCloudCoverVectorizer(CoreVectorizer):

    def vectorize(self, sample: WeatherForecast) -> List[float]:
        return super().vectorize(sample) +  [self._scale(sample.visibility, 50000),
                                             self._scale(sample.cloud_cover, 200)]

    def __str__(self):
        return "Core+Visibility+CloudVectorizer"


class PlusVisibilityFogCloudCoverVectorizer(CoreVectorizer):

    def vectorize(self, sample: WeatherForecast) -> List[float]:
        return super().vectorize(sample) +  [self._scale(sample.visibility, 50000),
                                             self._scale(sample.cloud_cover, 200),
                                             self._scale(sample.probability_for_fog, 100)]

    def __str__(self):
        return "Core+Visibility+Fog+CloudVectorizer"


class FullVectorizer(CoreVectorizer):

    def vectorize(self, sample: WeatherForecast) -> List[float]:
        return super().vectorize(sample) +  [self._scale(sample.visibility, 50000),
                                             self._scale(sample.cloud_cover, 200),
                                             self._scale(sample.probability_for_fog, 100),
                                             self._scale(sample.sunshine, 5000)]

    def __str__(self):
        return "FullVectorizer"


@dataclass(frozen=True)
class TrainReport:
    samples: List[LabelledWeatherForecast]


class TestReport:

    def __init__(self, train_report: TrainReport, validation_samples: List[LabelledWeatherForecast], predictions: List[int]):
        self.train_report = train_report
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
        if abs_values == 0:
            return 10000
        else:
            return round(sum(abs_values) / len(abs_values), 2)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        txt = "score:  " + str(self.score) + " (samples: " + str(len(self.train_report.samples)) + ")\n"
        txt += '{:14s}        {:14s} {:14s} {:14s} {:14s} {:14s}         {:10s} {:10s}           {:10s}\n'.format("time", "irradiance", "sunshine", "cloud_cover", "visibility", "proba.fog", "real", "predicted", "diff[%]")
        for i in range(0, len(self.validation_samples)):
            txt += '{:<14s}        {:<14d} {:<14d} {:<14d} {:<14d}  {:<14d}        {:<10d} {:<10d}           {:<10d}\n'.format(self.validation_samples[i].time.strftime("%d.%b  %H:%M"),
                                                                                                                               self.validation_samples[i].irradiance,
                                                                                                                               self.validation_samples[i].sunshine,
                                                                                                                               self.validation_samples[i].cloud_cover,
                                                                                                                               self.validation_samples[i].visibility,
                                                                                                                               self.validation_samples[i].probability_for_fog,
                                                                                                                               self.validation_samples[i].power_watt,
                                                                                                                               self.predictions[i],
                                                                                                                               int(self.__percent(self.validation_samples[i].power_watt, self.predictions[i])))
        txt = txt + "\nscore:  " + str(self.score)
        return txt


class Estimator:

    def __init__(self, vectorizer: Vectorizer):
        self.__clf = svm.SVC(kernel='poly') # it seems that the SVM approach produces good predictions. refer https://www.sciencedirect.com/science/article/pii/S136403212200274X?via%3Dihub and https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.221.4021&rep=rep1&type=pdf
        self.__vectorizer = vectorizer
        self.num_samples_last_train = 0
        self.num_covered_days_last_train = 0
        self.__score = None

    def usable_as_train_sample(self, sample: LabelledWeatherForecast) -> bool:
        return sample.irradiance > 0

    def clean_data(self, samples: List[LabelledWeatherForecast]) -> List[LabelledWeatherForecast]:
        seen = list()
        samples = list(filter(lambda sample: seen.append(sample.time_utc) is None if sample.time_utc not in seen else False, samples))
        samples = [sample for sample in samples if self.usable_as_train_sample(sample)]
        return samples

    def retrain(self, samples: List[LabelledWeatherForecast]) -> TrainReport:
        cleaned_samples = self.clean_data(samples)
        feature_vector_list = [self.__vectorizer.vectorize(sample) for sample in cleaned_samples]
        label_list = [sample.power_watt for sample in cleaned_samples]
        if len(set(label_list)) > 1:
            self.__clf.fit(feature_vector_list, label_list)
            self.num_samples_last_train = len(cleaned_samples)
            self.num_covered_days_last_train = len(set([sample.time.strftime("%Y.%m.%d") for sample in cleaned_samples]))
        return TrainReport(cleaned_samples)

    def test(self, samples: List[LabelledWeatherForecast], rounds: int = 10) -> TestReport:
        test_reports = []
        test_estimator = Estimator(self.__vectorizer)
        cleaned_samples = test_estimator.clean_data(samples)

        step_width = int(len(cleaned_samples) / rounds)
        if step_width < 1:
            step_width = 1
        for i in range(0, rounds):
            # split test data
            num_train_samples = int(len(cleaned_samples) * 0.65)
            train_samples = cleaned_samples[0: num_train_samples]
            validation_samples = cleaned_samples[num_train_samples:]

            # train and test
            train_report = test_estimator.retrain(train_samples)
            predicted = [test_estimator.predict(test_sample) for test_sample in validation_samples]
            test_reports.append(TestReport(train_report, validation_samples, predicted))

            cleaned_samples = cleaned_samples[-step_width:] + cleaned_samples[:-step_width]

        test_reports.sort(key=lambda report: report.score)
        median_report = test_reports[int(len(test_reports)*0.5)]
        self.__score = median_report.score
        return median_report

    def __str__(self):
        return "Estimator(vectorizer=" + str(self.__vectorizer) + "; deviation: " + ("unknown" if self.__score is None else str(round(self.__score, 1)) +"%") + "; trained with " + str(self.num_samples_last_train) + " samples; " + str(self.num_covered_days_last_train) + " days)"

    def predict(self, sample: WeatherForecast) -> int:
        if sample.irradiance > 0:
            feature_vector = self.__vectorizer.vectorize(sample)
            predicted = int(self.__clf.predict([feature_vector])[0])
            logging.debug(str(predicted) + " watt predicted for " + str(sample) + " (features: " + str(feature_vector) + ")")
            return predicted
        else:
            return 0
