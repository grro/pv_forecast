import logging
from typing import List
from pvpower.traindata import TrainSampleLog
from pvpower.estimator import Estimator, Vectorizer
from pvpower.forecast import WeatherForecast
from pvpower.tester import Tester


class CoreVectorizer(Vectorizer):

    def vectorize(self, sample: WeatherForecast) -> List[float]:
        vectorized = [self._scale(sample.utc_time.month, 12),
                      self._scale(int(self._utc_minutes_of_day(sample.utc_time)/10), int((24*60)/10)),
                      self._scale(sample.irradiance, 1000)]
        return vectorized


class PlusVisibilityVectorizer(Vectorizer):

    def vectorize(self, sample: WeatherForecast) -> List[float]:
        vectorized = [self._scale(sample.utc_time.month, 12),
                      self._scale(int(self._utc_minutes_of_day(sample.utc_time)/10), int((24*60)/10)),
                      self._scale(sample.irradiance, 1000),
                      self._scale(sample.visibility, 50000)]
        return vectorized


class PlusSunshineVectorizer(Vectorizer):

    def vectorize(self, sample: WeatherForecast) -> List[float]:
        vectorized = [self._scale(sample.utc_time.month, 12),
                      self._scale(int(self._utc_minutes_of_day(sample.utc_time)/10), int((24*60)/10)),
                      self._scale(sample.irradiance, 1000),
                      self._scale(sample.sunshine, 5000)]
        return vectorized


class PlusCloudCoverVectorizer(Vectorizer):

    def vectorize(self, sample: WeatherForecast) -> List[float]:
        vectorized = [self._scale(sample.utc_time.month, 12),
                      self._scale(int(self._utc_minutes_of_day(sample.utc_time)/10), int((24*60)/10)),
                      self._scale(sample.irradiance, 1000),
                      self._scale(sample.cloud_cover, 200)]
        return vectorized


class PlusFogVectorizer(Vectorizer):

    def vectorize(self, sample: WeatherForecast) -> List[float]:
        vectorized = [self._scale(sample.utc_time.month, 12),
                      self._scale(int(self._utc_minutes_of_day(sample.utc_time)/10), int((24*60)/10)),
                      self._scale(sample.irradiance, 1000),
                      self._scale(sample.probability_for_fog, 100)]
        return vectorized


class PlusVisibilitySunshineVectorizer(Vectorizer):

    def vectorize(self, sample: WeatherForecast) -> List[float]:
        vectorized = [self._scale(sample.utc_time.month, 12),
                      self._scale(int(self._utc_minutes_of_day(sample.utc_time)/10), int((24*60)/10)),
                      self._scale(sample.irradiance, 1000),
                      self._scale(sample.visibility, 50000),
                      self._scale(sample.sunshine, 5000)]
        return vectorized


class PlusVisibilityCloudCoverVectorizer(Vectorizer):

    def vectorize(self, sample: WeatherForecast) -> List[float]:
        vectorized = [self._scale(sample.utc_time.month, 12),
                      self._scale(int(self._utc_minutes_of_day(sample.utc_time)/10), int((24*60)/10)),
                      self._scale(sample.irradiance, 1000),
                      self._scale(sample.visibility, 50000),
                      self._scale(sample.cloud_cover, 200)]
        return vectorized


class PlusVisibilityFogVectorizer(Vectorizer):

    def vectorize(self, sample: WeatherForecast) -> List[float]:
        vectorized = [self._scale(sample.utc_time.month, 12),
                      self._scale(int(self._utc_minutes_of_day(sample.utc_time)/10), int((24*60)/10)),
                      self._scale(sample.irradiance, 1000),
                      self._scale(sample.visibility, 50000),
                      self._scale(sample.probability_for_fog, 100)]
        return vectorized



class PlusVisibilityFogCloudCoverVectorizer(Vectorizer):

    def vectorize(self, sample: WeatherForecast) -> List[float]:
        vectorized = [self._scale(sample.utc_time.month, 12),
                      self._scale(int(self._utc_minutes_of_day(sample.utc_time)/10), int((24*60)/10)),
                      self._scale(sample.irradiance, 1000),
                      self._scale(sample.visibility, 50000),
                      self._scale(sample.probability_for_fog, 100),
                      self._scale(sample.cloud_cover, 100)]
        return vectorized


class AllVectorizer(Vectorizer):

    def vectorize(self, sample: WeatherForecast) -> List[float]:
        vectorized = [self._scale(sample.utc_time.month, 12),
                      self._scale(int(self._utc_minutes_of_day(sample.utc_time)/10), int((24*60)/10)),
                      self._scale(sample.irradiance, 1000),
                      self._scale(sample.visibility, 50000),
                      self._scale(sample.probability_for_fog, 100),
                      self._scale(sample.cloud_cover, 100),
                      self._scale(sample.sunshine, 5000)]
        return vectorized


class VectorizerTester:

    def __init__(self, train_dir: str = None):
        trainlog = TrainSampleLog(train_dir)
        self.__samples = Estimator().clean_data(trainlog.all())
        self.__vectorizer_map = {
            "core": CoreVectorizer(),
            "+visibility": PlusVisibilityVectorizer(),
            "+sunshine": PlusSunshineVectorizer(),
            "+cloudcover": PlusCloudCoverVectorizer(),
            "+fog": PlusFogVectorizer(),
            "+visibility +sunshine": PlusVisibilitySunshineVectorizer(),
            "+visibility +cloudcover": PlusVisibilityCloudCoverVectorizer(),
            "+visibility +fog": PlusVisibilityFogVectorizer(),
            "+visibility +fog +cloudcover": PlusVisibilityFogCloudCoverVectorizer(),
            "+visibility +fog +cloudcover +sunshine": AllVectorizer(),
        }

    def report(self, num_rounds: int = 10) -> str:
        days = len(set([sample.utc_time.strftime("%Y.%m.%d")  for sample in self.__samples]))
        report = "tested with " + str(len(self.__samples)) + " cleaned samples (" + str(days) + " days; " + str(num_rounds) + " test rounds per variant)" + "\n"
        report += "VARIANT ................................. SCORE ....... DISTRIBUTION\n"
        for variant in self.__vectorizer_map.keys():
            test_reports = Tester(self.__samples).evaluate(Estimator(vectorizer=self.__vectorizer_map[variant]), rounds=num_rounds)
            median_report = test_reports[int(len(test_reports)*0.5)]
            score = str(round(median_report.score))
            distribution = str(round(test_reports[0].score)) + ", " + str(round(test_reports[1].score)) + ", " + str(round(test_reports[2].score)) + ", " + str(round(test_reports[3].score)) + ", ..., " + str(round(test_reports[int(len(test_reports)*0.5)].score)) + ", ..., " + str(round(test_reports[-4].score)) + ", " + str(round(test_reports[-3].score)) + ", " + str(round(test_reports[-2].score)) + ", " + str(round(test_reports[-1].score))
            report += variant + " " + "".join(["."] * (45 - (len(variant)+len(score)))) + " " + score + " ....... " + distribution + "\n"
            logging.info("variant " + variant + " analyzed")
        return report

