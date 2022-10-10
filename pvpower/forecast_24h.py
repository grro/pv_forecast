from datetime import datetime, timedelta
from typing import List, Dict, Optional
from pvpower.forecast import PvPowerForecast, LabelledWeatherForecast


class TimeFrame:

    def __init__(self, hourly_forecasts: List[LabelledWeatherForecast]):
        self.__hourly_forecasts = hourly_forecasts

    @property
    def start_time(self) -> datetime:
        return self.__hourly_forecasts[0].time

    @property
    def end_time(self) -> datetime:
        return self.__hourly_forecasts[-1].time + timedelta(minutes=59)

    @property
    def width_hours(self) -> int:
        return len(self.__hourly_forecasts)

    @property
    def hourly_power(self) -> List[int]:
        return [hourly_forecast.power_watt for hourly_forecast in self.__hourly_forecasts]

    @property
    def power_total(self) -> int:
        return sum(self.hourly_power)

    def __lt__(self, other):
        return self.start_time < other.start_time

    def __eq__(self, other):
        return self.start_time == other.start_time

    def __repr__(self):
        return self.__str__()

    def __str__(self) -> str:
        return self.__hourly_forecasts[0].time.strftime("%d.%m %H:%M") + " -> " + self.__hourly_forecasts[-1].time.strftime("%H") + ":59" + "; total: " + ", ".join([str(hourly_forecast.power_watt) for hourly_forecast in self.__hourly_forecasts])


class TimeFrames:

    def __init__(self, frames: List[TimeFrame]):
        self.__frames = sorted(frames, key=lambda frame: "{0:10d}".format(1000000-frame.power_total) + frame.start_time.strftime("%d.%m.%Y %H"), reverse=False)

    def filter(self, min_watt_per_hour: int):
        filtered_frames = [frame for frame in self.__frames if max(sorted(frame.hourly_power)) > min_watt_per_hour]
        return TimeFrames(filtered_frames)

    def empty(self) -> bool:
        return len(self.__frames) == 0

    def best(self) -> Optional[TimeFrame]:
        if len(self.__frames) > 0:
            return self.__frames[0]
        else:
            return None

    def second_best(self) -> Optional[TimeFrame]:
        if len(self.__frames) > 1:
            return self.__frames[1]
        else:
            return None

    def all(self) -> List[TimeFrame]:
        return list(self.__frames)


class Next24hours:

    def __init__(self, predictions: Dict[datetime, LabelledWeatherForecast]):
        self.predictions = predictions

    @staticmethod
    def of(pv_forecast: PvPowerForecast):
        now = datetime.strptime((datetime.now()).strftime("%d.%m.%Y %H"), "%d.%m.%Y %H")
        predictions = {}
        for weather_forecast in [pv_forecast.weather_forecast_service.forecast(prediction_time) for prediction_time in [now + timedelta(hours=i) for i in range(0, 40)]]:
            predicted_value = pv_forecast.predict_by_weather_forecast(weather_forecast)
            if predicted_value is not None:
                predictions[weather_forecast.time] = LabelledWeatherForecast.create(weather_forecast, predicted_value)
        return Next24hours(predictions)

    def __prediction_values(self) -> List[int]:
        return [forecast.power_watt for forecast in self.predictions.values() if forecast.time <= (datetime.now() + timedelta(hours=24))]

    def peek(self) -> int:
        return max(self.__prediction_values())

    def peek_time(self) -> datetime:
        peek_time = None
        peek_value = 0
        for dt in self.predictions.keys():
            forecast = self.predictions[dt]
            if forecast.power_watt > peek_value:
                peek_value = forecast.power_watt
                peek_time = dt
        return peek_time

    def power_total(self) -> int:
        return sum(self.__prediction_values())

    def frames(self, width_hours: int = 1) -> TimeFrames:
        frames = []
        times = list(self.predictions.keys())
        for offset_hour in range(0, 24+width_hours):
            forecasts = [self.predictions[times[idx]] for idx in range(offset_hour, offset_hour + width_hours)]
            frame = TimeFrame(forecasts)
            frames.append(frame)
        frames = [frame for frame in frames if frame.start_time <= datetime.now() + timedelta(hours=24)]
        return TimeFrames(frames)

    def __str__(self):
        txt = ""
        for time in list(self.predictions.keys())[:24]:
            txt += time.strftime("%d.%m %H:%M") + ": " + str(self.predictions[time].power_watt) +\
                   "  (irradiance=" + str(self.predictions[time].irradiance) + ", cloud_cover=" + str(self.predictions[time].cloud_cover) + ")\n"
        return txt


