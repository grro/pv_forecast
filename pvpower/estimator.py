import logging
import time
from dataclasses import dataclass
from datetime import datetime
from abc import ABC, abstractmethod
from sklearn import svm
from typing import List
from pvpower.weather_forecast import WeatherForecast
from pvpower.traindata import LabelledWeatherForecast, TrainData



class Vectorizer(ABC):

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
                      self._scale(sample.time_utc.hour, 24),
                      self._scale(sample.irradiance, 1000)]
        return vectorized

    def __str__(self):
        return "CoreVectorizer"


class SunshineVectorizer(Vectorizer):

    def vectorize(self, sample: WeatherForecast) -> List[float]:
        vectorized = [self._scale(sample.time_utc.month, 12),
                      self._scale(sample.time_utc.hour, 24),
                      self._scale(sample.sunshine, 5000)]
        return vectorized

    def __str__(self):
        return "SunshineVectorizer"


class SushinePlusCloudCoverVectorizer(SunshineVectorizer):

    def vectorize(self, sample: WeatherForecast) -> List[float]:
        return super().vectorize(sample) +  [self._scale(sample.cloud_cover_effective, 200)]

    def __str__(self):
        return "Sunshine+CloudVectorizer"



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
        return super().vectorize(sample) +  [self._scale(sample.cloud_cover_effective, 200)]

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
                                             self._scale(sample.cloud_cover_effective, 200)]

    def __str__(self):
        return "Core+Visibility+CloudVectorizer"


class PlusVisibilityFogCloudCoverVectorizer(CoreVectorizer):

    def vectorize(self, sample: WeatherForecast) -> List[float]:
        return super().vectorize(sample) +  [self._scale(sample.visibility, 50000),
                                             self._scale(sample.cloud_cover_effective, 200),
                                             self._scale(sample.probability_for_fog, 100)]

    def __str__(self):
        return "Core+Visibility+Fog+CloudVectorizer"


class FullVectorizer(CoreVectorizer):

    def vectorize(self, sample: WeatherForecast) -> List[float]:
        return super().vectorize(sample) +  [self._scale(sample.visibility, 50000),
                                             self._scale(sample.cloud_cover_effective, 200),
                                             self._scale(sample.probability_for_fog, 100),
                                             self._scale(sample.sunshine, 5000)]

    def __str__(self):
        return "FullVectorizer"


class Estimator(ABC):

    @abstractmethod
    def variant(self) -> str:
        pass

    @abstractmethod
    def date_last_train(self) -> datetime:
        pass

    @abstractmethod
    def num_samples_last_train(self) -> int:
        pass

    @abstractmethod
    def duration_sec_last_train(self) -> float:
        pass

    @abstractmethod
    def retrain(self, train_data: TrainData):
        pass

    @abstractmethod
    def predict(self, sample: WeatherForecast) -> int:
        pass


class DelegatingEstimator(Estimator):

    def __init__(self, estimator: Estimator):
        self._estimator = estimator

    def variant(self) -> str:
        return self._estimator.variant()

    def date_last_train(self) -> datetime:
        return self._estimator.date_last_train()

    def duration_sec_last_train(self) -> float:
        return self._estimator.duration_sec_last_train()

    def num_samples_last_train(self) -> int:
        return self._estimator.num_samples_last_train()

    def retrain(self, train_data: TrainData):
        self._estimator.retrain(train_data)

    def predict(self, sample: WeatherForecast) -> int:
        return self._estimator.predict(sample)

    def __str__(self):
        return str(self._estimator)


class ZeroIrradianceFilteringEstimator(Estimator):

    def retrain(self, train_data: TrainData):
        # special handling zero irradiance records
        samples = [sample for sample in train_data.samples if sample.irradiance > 0]
        self._do_retrain(TrainData(samples))

    @abstractmethod
    def _do_retrain(self, train_data: TrainData):
        pass

    def predict(self, sample: WeatherForecast) -> int:
        # special handling zero irradiance records. no irradiance means no power
        if sample.irradiance > 0:
            return self._do_predict(sample)
        else:
            return 0

    @abstractmethod
    def _do_predict(self, sample: WeatherForecast) -> int:
        pass


class SVMEstimator(ZeroIrradianceFilteringEstimator):

    def __init__(self, vectorizer: Vectorizer):
        self.__clf = svm.SVC(kernel='poly')   # it seems that the SVM approach produces good predictions. refer https://www.sciencedirect.com/science/article/pii/S136403212200274X?via%3Dihub and https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.221.4021&rep=rep1&type=pdf
        self.__vectorizer = vectorizer
        self.__date_last_train = datetime.fromtimestamp(0)
        self.__duration_last_train_sec = 0.0
        self.__num_samples_last_train = 0
        self.__num_covered_days_last_train = 0

    def variant(self) -> str:
        return type(self.__vectorizer).__name__

    def date_last_train(self) -> datetime:
        return self.__date_last_train

    def duration_sec_last_train(self) -> float:
        return self.__duration_last_train_sec

    def num_samples_last_train(self) -> int:
        return self.__num_samples_last_train

    def _do_retrain(self, train_data: TrainData):
        start = time.time()
        samples = train_data.samples
        feature_vector_list = [self.__vectorizer.vectorize(sample) for sample in samples]
        label_list = [sample.power_watt for sample in samples]
        if len(set(label_list)) > 1:
            self.__clf.fit(feature_vector_list, label_list)

            self.__date_last_train = datetime.now()
            self.__duration_last_train_sec = time.time() - start

            self.__num_samples_last_train = len(samples)
            self.__num_covered_days_last_train = len(set([sample.time.strftime("%Y.%m.%d") for sample in samples]))
            logging.debug("estimator has been trained " + str(self))
        else:
            logging.debug("estimator can not be trained. Insufficient train data")

    def _do_predict(self, sample: WeatherForecast) -> int:
        if self.__num_samples_last_train < 1:
            logging.warning("estimator has not been trained (insufficient train data available). returning 0")
            return 0
        else:
            feature_vector = self.__vectorizer.vectorize(sample)
            predicted = int(self.__clf.predict([feature_vector])[0])
            logging.debug(str(predicted) + " watt predicted for " + str(sample) + " (features: " + str(feature_vector) + ")")
            if predicted >= 0:
                return predicted
            else:
                logging.debug("predicted value is " + str(predicted) + " correct them to 0")
                return 0

    def __str__(self):
        return "SVMEstimator(vectorizer=" + str(self.__vectorizer) + "; trained with " + str(self.__num_samples_last_train) + \
               " samples; duration " +  str(round(self.__duration_last_train_sec, 3)) + " sec; age " + \
               str(datetime.now() - self.__date_last_train) + "; time range: " + str(self.__num_covered_days_last_train) + " days)"

