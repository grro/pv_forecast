import unittest
import tempfile
import os
import pytz
from random import randrange
from pvpower.mosmix import MosmixS, MemoryCachedMosmixLoader
from datetime import datetime, timedelta



class TestMosmix(unittest.TestCase):

    def test_load_from_web(self):
        mosmix = MemoryCachedMosmixLoader('N0677').get()
        tomorrow = datetime.now() + timedelta(days=1)
        irradiance = mosmix.rad1h(tomorrow)
        self.assertTrue(irradiance >= 0)
        #print("issue time utc:   " + str(mosmix.issue_time_utc))
        #print("issue time local: " + str(mosmix.issue_time))

    def test_save_and_restore(self):
        mosmix = MemoryCachedMosmixLoader('N0677').get()
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_file = os.path.join(tmp_dir, str(randrange(100000)) + ".temp")
            mosmix.save(temp_file)
            restored_mosmix = MosmixS.load(temp_file)
            self.assertEqual(mosmix.station_id, restored_mosmix.station_id)
            self.assertEqual(mosmix.issue_time, restored_mosmix.issue_time)
            self.assertEqual(mosmix.is_expired(), restored_mosmix.is_expired())
            time = datetime.now() + timedelta(days=5)
            self.assertEqual(mosmix.supports(time), restored_mosmix.supports(time))
            self.assertEqual(mosmix.vv(time), restored_mosmix.vv(time))
            os.remove(temp_file)

    def test_utc_local_time(self):
        mosmix = MemoryCachedMosmixLoader('N0677').get()

        print("cross-check with https://mosmix.de/online.html#/station/N0677/")
        print("time (local) ............. time (utc) .............. vv")
        now_local = datetime.now()
        for i in range(0, 10):
            tomorrow = datetime.strptime((datetime.now() + timedelta(days=1)).strftime("%d.%m.%Y %H") + ":00", "%d.%m.%Y %H:%M") + timedelta(hours=i)
            local = tomorrow.strftime("%d.%m.%Y %H:%M")
            vv = str(mosmix.vv(tomorrow))
            utc = tomorrow.astimezone(pytz.UTC).strftime("%d.%m.%Y %H:%M")
            print(local + " " + "".join(["."] * (25 - len(utc))) + " " + utc + " " + "".join(["."] * (15 - len(vv))) + " " + vv)


if __name__ == '__main__':
    unittest.main()