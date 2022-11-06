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
        return self.__hourly_forecasts[0].time.strftime("%H:%M") + "->" + self.__hourly_forecasts[-1].time.strftime("%H") + ":00" + "; expected power/hour: " + ", ".join([str(hourly_forecast.power_watt)  + " watt" for hourly_forecast in self.__hourly_forecasts])


class TimeFrames:

    def __init__(self, frames: List[TimeFrame]):
        self.__frames = sorted(frames, key=lambda frame: "{0:10d}".format(1000000-frame.power_total) + frame.start_time.strftime("%d.%m.%Y %H"), reverse=False)

    def filter(self, min_watt_per_hour: int):
        filtered_frames = [frame for frame in self.__frames if max(sorted(frame.hourly_power)) > min_watt_per_hour]
        return TimeFrames(filtered_frames)

    def empty(self) -> bool:
        return len(self.__frames) == 0

    def __len__(self):
        return len(self.__frames)

    def best(self) -> Optional[TimeFrame]:
        if len(self.__frames) > 0:
            return self.__frames[0]
        else:
            return None

    def pos(self, pos: int, dflt: TimeFrame = None) -> TimeFrame:
        if dflt is None:
            dflt = self.__frames[0]
        if pos < len(self.__frames):
            return self.__frames[pos]
        else:
            return dflt

    def all(self) -> List[TimeFrame]:
        return list(self.__frames)

    def __str__(self):
        txt = "start time ......... end time ........... pv power total ........... power per hour\n"
        for frame in sorted(self.__frames, key=lambda frame: frame.start_time, reverse=False):
            power = str(round(frame.power_total))
            power_per_hour = ", ".join([str(round(power)) for power in frame.hourly_power])
            txt += frame.start_time.strftime("%d %b, %H:%M") + "  ..... " + \
                   frame.end_time.strftime("%d %b, %H:%M") + " " + \
                   "".join(["."] * (15 - len(power))) + " " + power + " watt" + " " + \
                   "".join(["."] * (25 - len(power_per_hour))) + " " + power_per_hour + "\n"
        return txt



class Next24hours:

    def __init__(self, predicted_power: Dict[datetime, LabelledWeatherForecast]):
        self.predicted_power = predicted_power

    @staticmethod
    def __round_hour(dt: datetime) -> datetime:
        return (dt.replace(second=0, microsecond=0, minute=0, hour=dt.hour) +timedelta(hours=dt.minute//30))

    @staticmethod
    def of(pv_forecast: PvPowerForecast):
        now = datetime.strptime((datetime.now()).strftime("%d.%m.%Y %H") + ":00", "%d.%m.%Y %H:%S")
        predicted_power = {}
        for weather_forecast in [pv_forecast.weather_forecast_service.forecast(prediction_time) for prediction_time in [now + timedelta(hours=i) for i in range(0, 40)]]:
            if weather_forecast is not None:
                predicted_value = pv_forecast.predict_by_weather_forecast(weather_forecast)
                if predicted_value is not None:
                    predicted_power[Next24hours.__round_hour(weather_forecast.time)] = LabelledWeatherForecast.create(weather_forecast, predicted_value)
        return Next24hours(predicted_power)

    def __prediction_values(self) -> List[int]:
        return [forecast.power_watt for forecast in self.predicted_power.values() if forecast.time <= (datetime.now() + timedelta(hours=24))]

    def peek(self) -> int:
        return max(self.__prediction_values())

    def peek_time(self) -> datetime:
        peek_time = None
        peek_value = 0
        for dt in self.predicted_power.keys():
            forecast = self.predicted_power[dt]
            if forecast.power_watt > peek_value:
                peek_value = forecast.power_watt
                peek_time = dt
        return peek_time

    def power_total(self) -> int:
        return sum(self.__prediction_values())

    def frames(self, width_hours: int = 1) -> TimeFrames:
        frames = []
        times = list(self.predicted_power.keys())
        for offset_hour in range(0, 24+width_hours):
            forecasts = [self.predicted_power[times[idx]] for idx in range(offset_hour, offset_hour + width_hours)]
            frame = TimeFrame(forecasts)
            frames.append(frame)
        frames = [frame for frame in frames if frame.start_time <= (datetime.now() + timedelta(hours=24))]
        return TimeFrames(frames)

    def __str__(self):
        txt = "time ................ pv power ..... irradiance ....... sunshine .... visibility .... fog probab. .... cloud cover\n"
        for time in list(self.predicted_power.keys())[:24]:
            power = str(round(self.predicted_power[time].power_watt))
            irradiance = str(round(self.predicted_power[time].irradiance))
            visibility = str(round(self.predicted_power[time].visibility))
            sunshine = str(round(self.predicted_power[time].sunshine))
            probability_for_fog = str(round(self.predicted_power[time].probability_for_fog))
            cloud_cover = str(round(self.predicted_power[time].cloud_cover))
            txt += time.strftime("%d %b, %H:%M") + " " + \
                   "".join(["."] * (10 - len(power))) + " " + power + " watt " + \
                   "".join(["."] * (15 - len(irradiance))) + " " + irradiance + " " + \
                   "".join(["."] * (15 - len(sunshine))) + " " + sunshine + \
                   "".join(["."] * (15 - len(visibility))) + " " + visibility + " " + \
                   "".join(["."] * (15 - len(probability_for_fog))) + " " + probability_for_fog +  " " + \
                   "".join(["."] * (15 - len(cloud_cover))) + " " + cloud_cover + "\n"
        return txt


