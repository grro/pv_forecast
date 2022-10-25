# photovoltaic power forecast

pv_forecast provides a set of artifacts to obtain PV solar power forecast. The library uses machine learning approaches to perform forecasts.
To get appropriated results, real measured PV power values must be delivered, periodically. Internally, the library make use of [DWD](https://dwd-geoportal.de/products/G_FJM/) weather forecast data.

**Installing the library**

To install this software you may use [PIP](https://realpython.com/what-is-pip/) package manager such as shown below
```
sudo pip install pvpower
```

**Using the library**

After this installation you should configure the library with your environment parameters.
You have to set the closest DWD station id of our PV system location. Refer [DWD station list](https://www.dwd.de/DE/leistungen/met_verfahren_mosmix/mosmix_stationskatalog.cfg?view=nasPublication&nn=16102) to select the proper station id.     
```
from pvpower.forecast import PvPowerForecast

dwd_station_id = 'L160'
pv_power_forecast = PvPowerForecast(dwd_station_id)
```

To get a power forecast, the predict method has to be called
```
tomorrow = datetime.now() + timedelta(days=1)
power_watt_tomorrow = forecast.predict(tomorrow)
```

**Train the library with real measurements**

It is essential that the PvPowerForecast library is provided with real measured PV values of our PV system. 
The provided real data is used to adapt the internal machine learning engine to your specific environment. 
Providing technical parameters of your PV system such as installed power or cardinal direction is not required. 
The **library is self-learning**.

```
# please provide the real measured PV power value periodically. 
# The period should be between 1 minute and 15 minutes.

while True:
    real_pv_power_watt = ...read real PV power ...
    pv_power_forecast.add_current_power_reading(real_pv_power_watt)
    time.sleep(60)
```
The provided real measurements will be stored internally on disc and be used to update the internal prediction model. 
Please consider, that more accurate forecast predictions require collecting real PV measurements for at least 2 weeks, typically. 
Do not stop providing measurements, even though the prediction becomes better and better. 
You may use a periodic job to provide the real PV values

**Energy management system support**

The basic functionality of this library is to support photovoltaic power forecast. However, to maximize the yield 
of your PV system, your home appliances such as a dishwasher or laundry machines should operate only in periods when 
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
In the example below the frames will be filtered considering a hypothetical basic electricity consumption of 350 watt per hour. Time frames will be considered only, 
if the solar power is higher than the expected basic electricity consumption. 
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