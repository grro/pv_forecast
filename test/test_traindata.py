import unittest
import gzip
import tempfile
from os import path
from pvpower.traindata import TrainSampleLog, LabelledWeatherForecast
from datetime import datetime



class TestTrainSampleLog(unittest.TestCase):

    def test_uct_format(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            log = TrainSampleLog(tmpdirname)
            dt1 = datetime.strptime("2021.11.30T13:30", "%Y.%m.%dT%H:%M")
            log.append(LabelledWeatherForecast(dt1, 100, 200, 300, 400, 500, 99))

            with gzip.open(path.join(tmpdirname, TrainSampleLog.FILENAME), 'rb') as f:
                lines = [raw_line.decode('UTF-8').strip() for raw_line in f.readlines()]
                self.assertEqual(1, len(lines))
                self.assertEqual('2021-11-30T12:30:00+00:00;99;100;200;300;400;500', lines[0])

            traindata = log.all()
            self.assertEqual(1, len(traindata.samples))
            self.assertEqual(dt1, traindata.samples[0].time)


if __name__ == '__main__':
    unittest.main()