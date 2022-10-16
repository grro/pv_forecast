import logging
import time
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Optional
from sklearn import svm
from typing import List
from pvpower.weather_forecast import WeatherForecast
from pvpower.traindata import LabelledWeatherForecast



class Vectorizer(ABC):

    @abstractmethod
    def vectorize(self, sample: WeatherForecast) -> List[float]:
        pass

    def _scale(self, value: int, max_value: int, digits=1) -> float:
        if value == 0:
            return 0
        else:
            return round(value * 100 / max_value, digits)


class BasicVectorizer(Vectorizer):

    def vectorize(self, sample: WeatherForecast) -> List[float]:
        vectorized = [self._scale(sample.time.month, 12),
                      self._scale((sample.time.hour*60) + (int(sample.time.minute/15) * 15), 24*60),
                      self._scale(sample.irradiance, 1000),
                      self._scale(sample.visibility, 50000)]
        #logging.info(sample.time.strftime("%d.%m.%Y %H:%M") + ";" + str(sample.irradiance) + "   ->   " + str(vectorized))
        return vectorized

    def __str__(self):
        return "BasicVectorizer(month,hour,irradiance)"


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
            self.__vectorizer = BasicVectorizer()
        else:
            self.__vectorizer = vectorizer
        self.__num_samples_last_train = 0

    def clean_data(self, samples: List[LabelledWeatherForecast]) -> List[LabelledWeatherForecast]:
        seen = list()
        samples = list(filter(lambda sample: seen.append(sample.time) is None if sample.time not in seen else False, samples))
        samples = [sample for sample in samples if sample.irradiance > 0]
        return samples

    def retrain(self, samples: List[LabelledWeatherForecast]) -> TrainReport:
        cleaned_samples = self.clean_data(samples)
        num_samples = len(cleaned_samples)
        if self.__num_samples_last_train != num_samples:
            if num_samples < 2:
                logging.warning("just " + str(len(cleaned_samples)) + " samples with irradiance > 0 are available. At least 2 samples are required")
            else:
                feature_vector_list = [self.__vectorizer.vectorize(sample) for sample in cleaned_samples]
                label_list = [sample.power_watt for sample in cleaned_samples]
                if len(set(label_list)) > 1:
                    self.__clf.fit(feature_vector_list, label_list)
                    self.__num_samples_last_train = num_samples
        return TrainReport(cleaned_samples)

    def __str__(self):
        return "Model vectorizer=" + str(self.__vectorizer) + " trained with " + str(self.__num_samples_last_train) + " cleaned samples"

    def predict(self, sample: WeatherForecast) -> Optional[int]:
        try:
            if sample.irradiance > 0:
                feature_vector = self.__vectorizer.vectorize(sample)
                #print(feature_vector)
                predicted = self.__clf.predict([feature_vector])[0]
                return int(predicted)
            else:
                return 0
        except Exception as e:
            logging.warning("error occurred predicting " + str(sample) + " " + str(e))
            return None
