from typing import List
from pvpower.estimator import Estimator
from pvpower.traindata import TrainData



def test(estimator: Estimator, train_data: TrainData):
    return TestReport(estimator, train_data)



class TestReport:

    def __init__(self, estimator: Estimator, train_data: TrainData):
        example_data, validation_data = train_data.split()
        self.validation_samples  = [reocrd for reocrd in validation_data.samples]
        estimator.retrain(example_data)
        self.predictions = [estimator.predict(validation_record) for validation_record in self.validation_samples]

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
        if len(abs_values) == 0:
            return 10000
        else:
            return round(sum(abs_values) / len(abs_values), 2)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        txt = "score:  " + str(self.score) + " (samples: " + str(len(self.validation_samples)) + ")\n"
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
