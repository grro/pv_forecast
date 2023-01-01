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

    def __percent(self, real, predicted):
        if real == 0 and predicted == 0:
            return 0
        elif real == 0 or predicted == 0:
            return 10000
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
                    diff = 10000
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
        if len(abs_values) == 0:
            return 10000
        else:
            return round(sum(abs_values) / len(abs_values), 2)

    def __lt__(self, other):
        return self.score < other.score

    def __eq__(self, other):
        return self.score == other.score

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        txt = str(self.estimator) + "\n"
        txt += "derivation:  " + str(self.score) + " \n"
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

        train_runs = []
        for estimator in estimators:
            runs = sorted([TrainRun(estimator, trainData.rotated(i * 23)) for i in range(0, 5)])
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
