import logging
import pickle
from os import path, makedirs
from os.path import exists
from appdirs import user_cache_dir
from threading import Thread, Lock
from datetime import datetime, timedelta
from pvpower.weather_forecast import WeatherForecast
from pvpower.traindata import TrainSampleLog, TrainData
from pvpower.estimator import Estimator, DelegatingEstimator, SVMEstimator, FullVectorizer
from pvpower.trainingcenter import TrainingCenter


class AutoRefreshingEstimator(DelegatingEstimator):

    def __init__(self, train_log: TrainSampleLog):
        self.__train_log = train_log
        self.__lock = Lock()
        self.__training_center = TrainingCenter()
        self.__date_last_retrain_initiated = datetime.fromtimestamp(0)
        super().__init__(AutoRefreshingEstimator.__load())

    def predict(self, sample: WeatherForecast) -> int:
        try:
            return super().predict(sample)
        finally:
            try:
                retrain_period_days = 1 + (self.duration_sec_last_train() * 7 * 24)  # min 1 day + 7 days per 1 sec traintime
                with self.__lock:
                    if datetime.now() > (self.__date_last_retrain_initiated + timedelta(hours=int(retrain_period_days*24))):
                        self.__date_last_retrain_initiated = datetime.now()
                        Thread(target=self.retrain, args=(self.__train_log.all(),), daemon=True).start()
            except Exception as e:
                logging.warning("error occurred checking last train duration of estimator", e)
                Thread(target=self.retrain, args=(self.__train_log.all(),), daemon=True).start()

    def retrain(self, train_data: TrainData):
        self._estimator = self.__training_center.new_estimator(train_data)
        self.__store(self._estimator)

    @staticmethod
    def __store(estimator: Estimator):
        try:
            with open(AutoRefreshingEstimator.__filename(), 'wb') as file:
                pickle.dump(estimator, file)
        except Exception as e:
            logging.warning("error occurred storing pickled estimator " + str(e))

    @staticmethod
    def __load() -> Estimator:
        estimator = SVMEstimator(FullVectorizer())
        if exists(AutoRefreshingEstimator.__filename()):
            try:
                with open(AutoRefreshingEstimator.__filename(), 'rb') as file:
                    estimator = pickle.load(file)
                    logging.debug("estimator " + str(estimator) + " loaded from pickle file (" + AutoRefreshingEstimator.__filename() + ")")
            except Exception as e:
                logging.warning("error occurred loading estimator " + str(e))
        return estimator

    @staticmethod
    def __filename():
        cache_dir = user_cache_dir("pv_forecast", appauthor=False)
        if not path.exists(cache_dir):
            makedirs(cache_dir)
        return path.join(cache_dir, 'Estimator.p')

    def __str__(self):
        return str(self._estimator)

