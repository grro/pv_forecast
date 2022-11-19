import unittest
from datetime import datetime, timedelta
from pvpower.weather_forecast import WeatherStation


class TestWeatherForecast(unittest.TestCase):

    def test_date_is_not_timezone_aware(self):
        station = WeatherStation('N0677')
        self.assertIsNone(station.forcast_to().tzinfo)
        self.assertIsNone(station.forcast_from().tzinfo)
        station.forecast()

    def test_forecast(self):
        station = WeatherStation('N0677')
        tomorrow = datetime.now()+ timedelta(days=1)
        forcast = station.forecast(tomorrow)
        self.assertEqual(tomorrow.strftime("%d.%m.%Y %H:%M"), forcast.time.strftime("%d.%m.%Y %H:%M"))
        self.assertIsNone(forcast.time.tzinfo)
        self.assertIsNotNone(forcast.time_utc.tzinfo)
        #print("utc:    " + str(forcast.time_utc))
        #print("local:  " + str(forcast.time))




if __name__ == '__main__':
    unittest.main()