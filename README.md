# photovoltaic power forecast

pv_forecast provides a set of classes to obtain PV solar power forecast. Internally this library uses machine learning to perform PV power forecasts.
To get appropriated results, real measured PV power values must be provided, periodically. Additionally, the library uses weather forecast data of [DWD](https://dwd-geoportal.de/products/G_FJM/) to perform accurate PV power forecasts.

To install this software you may use [PIP](https://realpython.com/what-is-pip/) package manager such as shown below

**PIP approach**
```
sudo pip install pvpower
```

After this installation you should configure the library with your environment parameters.
You have to set the closest DWD station id of the location of our PV system. To find the proper station id refer [DWD station list](https://www.dwd.de/DE/leistungen/met_verfahren_mosmix/mosmix_stationskatalog.cfg?view=nasPublication&nn=16102)     
```
from pvpower.forecast import PvPowerForecast

dwd_station_id = 'L160'
pv_power_forecast = PvPowerForecast(dwd_station_id)
```

**Train with real measurements**

It is essential that the PvPowerForecast library will be provided with real measured PV values of our PV system. 
The provided real data is used as train data to adapt the internal prediction model on your environment. 
Providing additional technical parameters of your PV system such as installed power or cardinal direction is not required. This **libray is self-learning**.
```
# please provide the real measured PV power value periodically. The period should be between 1 minute and 10 minutes.
while True:
    real_pv_power_watt = ...read real PV power ...
    pv_power_forecast.current_power_reading(real_pv_power_watt)
    time.sleep(60)
```
The provided train data will be store internally on disc and be used to update the internal prediction model. Please consider, that more accurate forecast predictions require collecting real PV power data for at least 2 weeks. Do not stop providing real PV power data, even though the prediction becomes better and better. You may use a periodic job to provide the real PV values

**PV power forecast**

To get the power forecast the power method has to be called 
```
tomorrow = datetime.now() + timedelta(days=1)
predicted_pv_power_watt_tomorrow = forecast.predict(tomorrow)
```

**Energy management system support**

The basic functionality of this library is to support photovoltaic power forecast. However, to maximize the yield 
of your PV system, home appliances such as a dishwasher or laundry machines should operate only in periods when 
your PV system is delivering sufficient solar power. To manage this, the *Next24hours* convenience class can be used as shown below 
```
from pvpower.forecast import PvPowerForecast
from pvpower.forecast_24h import Next24hours

power_forecast = PvPowerForecast('L160')
next24h = Next24hours.of(power_forecast)
peek_watt = next24h.peek()
...
```

To start your home appliance such as a dishwasher at the right time you may query the available execution time frames. 
In the example below the frames will be filtered considering the basic electricity consumption. Time frames will be considered only, 
if the solar power is higher than the basic electricity consumption. In the example an average basic consumption of 350 watt is expected.
Based on the resulting time frames the best one is used to start the home appliance in a delayed way.  
```
...
pogram_duration_hours = 3
best_time_frame = next24h.frames(width_hours=pogram_duration_hours).filter(min_watt_per_hour=350).best()
if best_time_frame is None:
    #.. start now (no sufficient solar power next 24h)
else:
    # .. start delayed when best window is reached
    start_time = best_time_frame.start_time
    ...
```