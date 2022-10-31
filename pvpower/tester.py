from typing import List
from pvpower.traindata import LabelledWeatherForecast
from pvpower.estimator import Estimator, TrainReport


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


class Tester:

    def __init__(self, samples: List[LabelledWeatherForecast]):
        self.__samples = samples

    def evaluate(self, estimator: Estimator, rounds: int = 10) -> List[TestReport]:
        test_reports = []
        samples = estimator.clean_data(self.__samples)

        step_width = int(len(samples) / rounds)
        if step_width < 1:
            step_width = 1
        for i in range(0, rounds):
            # split test data
            num_train_samples = int(len(samples) * 0.7)
            train_samples = samples[0: num_train_samples]
            validation_samples = samples[num_train_samples:]

            # train and test
            train_report = estimator.retrain(train_samples)
            predicted = [estimator.predict(test_sample) for test_sample in validation_samples]
            test_reports.append(TestReport(train_report, validation_samples, predicted))

            samples = samples[-step_width:] + samples[:-step_width]

        test_reports.sort(key=lambda report: report.score)
        return test_reports
