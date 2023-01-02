import logging
from typing import List
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

    def __derivation(self, real, predicted):
        if abs(real - predicted) < 10: # ignore diff < 10
            return 0
        elif real == 0:
            return predicted
        elif predicted == 0:
            return real
        else:
            return real-predicted


    @property
    def score(self) -> float:
        derivation_total = []
        for i in range(len(self.validation_samples)):
            if self.validation_samples[i].irradiance == 0:
                continue
            else:
                predicted = self.predictions[i]
            diff = self.__derivation(self.validation_samples[i].power_watt, predicted)
            derivation_total.append(diff)

        derivations = sorted([abs(value) for value in derivation_total])
        ignore_size = int(len(derivations) * 0.05)
        if ignore_size <= 0:
            ignore_size = 1
        derivations = derivations[ignore_size:-ignore_size]
        if len(derivations) == 0:
            return 999999
        else:
            return round(sum(derivations) / len(derivations), 2)

    def __lt__(self, other):
        return self.score < other.score

    def __eq__(self, other):
        return self.score == other.score

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        max_lines = 1000
        txt = str(self.estimator) + "\n"
        txt += "derivation:  " + str(self.score) + " \n"
        txt += '{:14s}        {:14s} {:14s} {:14s} {:14s} {:14s}         {:10s} {:10s}           {:10s}\n'.format("time", "irradiance", "sunshine", "cloud_cover", "visibility", "proba.fog", "real", "predicted", "derivation")
        for i in range(0, len(self.validation_samples)):
            if i > max_lines:
                txt += '....'
                break
            txt += '{:<14s}        {:<14d} {:<14d} {:<14d} {:<14d}  {:<14d}        {:<10d} {:<10d}           {:<10d}\n'.format(self.validation_samples[i].time.strftime("%d.%b  %H:%M"),
                                                                                                                               self.validation_samples[i].irradiance,
                                                                                                                               self.validation_samples[i].sunshine,
                                                                                                                               self.validation_samples[i].cloud_cover_effective,
                                                                                                                               self.validation_samples[i].visibility,
                                                                                                                               self.validation_samples[i].probability_for_fog,
                                                                                                                               self.validation_samples[i].power_watt,
                                                                                                                               self.predictions[i],
                                                                                                                               int(self.__derivation(self.validation_samples[i].power_watt, self.predictions[i])))
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

        rounds = 6
        train_runs = []
        for estimator in estimators:
            runs = sorted([TrainRun(estimator, trainData.rotated(round(i * 100 / rounds))) for i in range(0, rounds)])
            cleaned_runs = [run for run in runs if run.score < 10000]
            if len(cleaned_runs) > 0:
                runs = cleaned_runs
            train_runs.append(runs[int(len(runs)*0.5)])

        train_runs = sorted(train_runs)
        for run in train_runs:
            logging.debug(run)
        best_train = train_runs[0]
        logging.info("new estimator trained \n" + str(best_train))
        return best_train.estimator
