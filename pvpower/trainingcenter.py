import logging
from typing import List
from statistics import mean
from pvpower.estimator import Estimator, SVMEstimator, FullVectorizer, CoreVectorizer, SunshineVectorizer, PlusVisibilityVectorizer, SushinePlusCloudCoverVectorizer, PlusSunshineVectorizer, PlusCloudCoverVectorizer, PlusVisibilitySunshineVectorizer, PlusVisibilityCloudCoverVectorizer, PlusVisibilityFogCloudCoverVectorizer
from pvpower.traindata import TrainData


class TrainRun:

    def __init__(self, estimator: Estimator, train_data: TrainData):
        example_data, validation_data = train_data.split()
        self.validation_samples  = [record for record in validation_data.samples]
        self.estimator = estimator
        self.estimator.retrain(example_data)
        self.predictions = [estimator.predict(validation_record) for validation_record in self.validation_samples]
        self.estimator.retrain(train_data)  # retrain with all data

    def __score(self, real, predicted)-> int:
        return round(abs(real - predicted) / 10)*10

    @property
    def score(self) -> float:
        scores = []
        for i in range(len(self.validation_samples)):
            real = self.validation_samples[i].power_watt
            predicted = self.predictions[i]
            if real == 0 and predicted == 0:   # do not waste the total score by true zero predictions
                continue
            scores.append(self.__score(real, predicted))
        scores_without_outliners = self.__without_outliners(scores, 0.1)
        return round(mean(scores_without_outliners), 2)

    @staticmethod
    def __without_outliners(scores: List[int], percent: float) -> List[int]:
        scores = sorted(list(scores))
        ignore_size = int(len(scores) * percent)
        if ignore_size <= 0:
            ignore_size = 1
        return scores[ignore_size:-ignore_size]

    def __lt__(self, other):
        return self.score < other.score

    def __eq__(self, other):
        return self.score == other.score

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        num_considered = 0
        max_lines = 1000
        txt = str(self.estimator) + "\n"
        txt += "score:  " + str(self.score) + " (smaller is better)\n"
        txt += '{:25s}   {:10s} {:10s}    {:10s}          {:14s} {:14s} {:14s} {:14s} {:14s}         \n'.format("time", "real", "predicted", "score", "irradiance", "sunshine", "cloud_cover", "visibility", "proba.fog")
        for i in range(0, len(self.validation_samples)):
            if self.validation_samples[i].irradiance == 0 and self.predictions[i] == 0:
                continue
            if num_considered > max_lines:
                txt += '....'
                break
            num_considered += 1
            txt += '{:<25s}   {:<10d} {:<10d}    {:<10s}          {:<14d} {:<14d} {:<14s} {:<14d}  {:<14d}        \n'.format(self.validation_samples[i].time.strftime("%d.%b %H:%M") + " (" + self.validation_samples[i].time_utc.strftime("%H:%M") + " utc)",
                                                                                                                               self.validation_samples[i].power_watt,
                                                                                                                               self.predictions[i],
                                                                                                                               "[" + str(self.__score(self.validation_samples[i].power_watt, self.predictions[i])) + "]",
                                                                                                                               self.validation_samples[i].irradiance,
                                                                                                                               self.validation_samples[i].sunshine,
                                                                                                                               str(self.validation_samples[i].cloud_cover_effective) + "%",
                                                                                                                               self.validation_samples[i].visibility,
                                                                                                                               self.validation_samples[i].probability_for_fog)
        return txt



class TrainingCenter:

    def new_estimator(self, trainData: TrainData) -> Estimator:
        estimators = [SVMEstimator(CoreVectorizer()),
                      SVMEstimator(FullVectorizer()),
                      SVMEstimator(SunshineVectorizer()),
                      SVMEstimator(SushinePlusCloudCoverVectorizer()),
                      SVMEstimator(PlusVisibilityVectorizer()),
                      SVMEstimator(PlusSunshineVectorizer()),
                      SVMEstimator(PlusCloudCoverVectorizer()),
                      SVMEstimator(PlusVisibilitySunshineVectorizer()),
                      SVMEstimator(PlusVisibilityCloudCoverVectorizer()),
                      SVMEstimator(PlusVisibilityFogCloudCoverVectorizer())]

        logging.info("train estimators with " + str(len(trainData.samples)) + " samples")
        rounds = 6
        train_runs = []
        for estimator in estimators:
            runs = sorted([TrainRun(estimator, trainData.rotated(round(i * 100 / rounds))) for i in range(0, rounds)])
            cleaned_runs = [run for run in runs if run.score < 10000]
            if len(cleaned_runs) > 0:
                runs = cleaned_runs
            train_runs.append(runs[int(len(runs)*0.5)])

        train_runs = sorted(train_runs, reverse=True)
        for run in train_runs:
            logging.debug(run)
        best_train = train_runs[-1]
        logging.info("new estimator trained \n" + str(best_train))
        return best_train.estimator
