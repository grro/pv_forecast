import logging
import pytz
from dataclasses import dataclass
from datetime import datetime, timedelta
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


class FullVectorizer(Vectorizer):

    def vectorize(self, sample: WeatherForecast) -> List[float]:
        vectorized = [self._scale(sample.utc_time.month, 12),
                      self._scale(int(self._utc_minutes_of_day(sample.utc_time)/10), int((24*60)/10)),
                      self._scale(sample.irradiance, 1000),
                      self._scale(sample.visibility, 50000),
                      self._scale(sample.probability_for_fog, 100),
                      self._scale(sample.cloud_cover, 100),
                      self._scale(sample.sunshine, 5000)]
        return vectorized

    def __str__(self):
        return "FullVectorizer"


@dataclass(frozen=True)
class TrainReport:
    samples: List[LabelledWeatherForecast]



class Estimator:

    def __init__(self, classifier= None, vectorizer: Vectorizer = None):
        # it seems that the SVM approach produces good predictions
        # refer https://www.sciencedirect.com/science/article/pii/S136403212200274X?via%3Dihub and https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.221.4021&rep=rep1&type=pdf
        if classifier is None:
            self.__clf = svm.SVC(kernel='poly')
        else:
            self.__clf = classifier
        if vectorizer is None:
            self.__vectorizer = FullVectorizer()
        else:
            self.__vectorizer = vectorizer
        self.num_samples_last_train = 0
        # initialize with dummy data
        self.retrain([LabelledWeatherForecast(datetime.now(), 1, 0, 0, 0, 0, 0), LabelledWeatherForecast(datetime.now() - timedelta(days=1), 1, 0, 0, 0, 0, 1)])

    def usable_as_train_sample(self, sample: LabelledWeatherForecast) -> bool:
        return sample.irradiance > 0 \
               and sample.sunshine is not None \
               and sample.visibility is not None

    def clean_data(self, samples: List[LabelledWeatherForecast]) -> List[LabelledWeatherForecast]:
        seen = list()
        samples = list(filter(lambda sample: seen.append(sample.utc_time) is None if sample.utc_time not in seen else False, samples))
        samples = [sample for sample in samples if self.usable_as_train_sample(sample)]
        return samples

    def retrain(self, samples: List[LabelledWeatherForecast]) -> TrainReport:
        cleaned_samples = self.clean_data(samples)
        feature_vector_list = [self.__vectorizer.vectorize(sample) for sample in cleaned_samples]
        label_list = [sample.power_watt for sample in cleaned_samples]
        if len(set(label_list)) > 1:
            self.__clf.fit(feature_vector_list, label_list)
            self.num_samples_last_train = len(cleaned_samples)
        return TrainReport(cleaned_samples)

    def __str__(self):
        return "Model vectorizer=" + str(self.__vectorizer) + " trained with " + str(self.num_samples_last_train) + " cleaned samples"

    def predict(self, sample: WeatherForecast) -> int:
        if sample.irradiance > 0:
            feature_vector = self.__vectorizer.vectorize(sample)
            predicted = self.__clf.predict([feature_vector])[0]
            return int(predicted)
        else:
            return 0
