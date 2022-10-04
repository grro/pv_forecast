from datetime import datetime, timedelta
from typing import List
from pvpower.pv_forecast_service import PvPowerForecast, LabelledWeatherForecast


class Frame:

    def __init__(self, min_power_watt: int, hourly_forecasts: List[LabelledWeatherForecast]):
        self.min_power_watt = min_power_watt
        self.__hourly_forecasts = hourly_forecasts
        self.__surplus = []
        for hourly_forecast in hourly_forecasts:
            if hourly_forecast.power_watt > min_power_watt:
                self.__surplus.append(hourly_forecast.power_watt - min_power_watt)

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
    def surplus_total(self):
        return sum(self.__surplus)

    @property
    def power_total(self) -> int:
        return sum([hourly_forecast.power_watt for hourly_forecast in self.__hourly_forecasts])

    def __repr__(self):
        return self.__str__()

    def __str__(self) -> str:
        return self.__hourly_forecasts[0].time.strftime("%H:%M") + " -> " + self.__hourly_forecasts[-1].time.strftime("%H") + ":59" + ":  surplus=" + str(self.__surplus) + " (total " + ", ".join([str(hourly_forecast.power_watt) for hourly_forecast in self.__hourly_forecasts]) + ")"


class ForecastNext24hours:

    def __init__(self, pv_forecast: PvPowerForecast):
        now = datetime.strptime((datetime.now()).strftime("%d.%m.%Y %H"), "%d.%m.%Y %H")
        self.predictions = {}
        for weather_forecast in [pv_forecast.weather_forecast_service.forecast(prediction_time) for prediction_time in [now + timedelta(hours=i) for i in range(0, 40)]]:
            predicted_value = pv_forecast.predict_by_weather_forecast(weather_forecast)
            if predicted_value is not None:
                self.predictions[weather_forecast.time] = LabelledWeatherForecast.create(weather_forecast, predicted_value)

    def __prediction_values(self) -> List[int]:
        return [forecast.power_watt for forecast in self.predictions.values() if forecast.time <= (datetime.now() + timedelta(hours=24))]

    def peek(self) -> int:
        return max(self.__prediction_values())

    def power_at(self, time: datetime) -> int:
        for weather_forecast in self.predictions.keys():
            if weather_forecast.time.strftime("%d.%m.%Y %H") == time.strftime("%d.%m.%Y %H"):
                return self.predictions[weather_forecast][1]
        return 0

    def frame_extra_power(self, base_power_watt: int, width_hours: int = 1) -> List[Frame]:
        frames = []
        times = list(self.predictions.keys())
        for offset_hour in range(0, 24+width_hours):
            forecasts = [self.predictions[times[idx]] for idx in range(offset_hour, offset_hour + width_hours)]
            frame = Frame(base_power_watt, forecasts)
            frames.append(frame)
        frames = [slot for slot in frames if slot.surplus_total > 0]
        frames = [slot for slot in frames if slot.start_time <= datetime.now() + timedelta(hours=24)]
        frames = sorted(frames, key=lambda slot: slot.surplus_total, reverse=True)
        return frames

    def __str__(self):
        txt = ""
        for time in list(self.predictions.keys())[:24]:
            txt += time.strftime("%d.%m %H:%M") + ": " + str(self.predictions[time].power_watt) +\
                   "  (irradiance=" + str(self.predictions[time].irradiance) + ", cloud_cover=" + str(self.predictions[time].cloud_cover) + ")\n"
        return txt


