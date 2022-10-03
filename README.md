# photovoltaic power forecast

pv_forecast provides a set of classes to obtain PV power forecast. Internally this library uses machine learning to perform PV power forecasts.
To get appropriated results, real measured PV power values must be provided, periodically. Additionally, the library uses weather forecast data of [DWD](https://dwd-geoportal.de/products/G_FJM/) to perform accurate PV power forecasts.

To install this software you may use [PIP](https://realpython.com/what-is-pip/) package manager such as shown below

**PIP approach**
```
sudo pip install pv_forecast
```

After this installation you should configure the library with your environment parameters.
You have to set the closest DWD station id of the location of our PV system. To find the proper station id refer [DWD station list](https://www.dwd.de/DE/leistungen/met_verfahren_mosmix/mosmix_stationskatalog.cfg?view=nasPublication&nn=16102)     
```
dwd_station_id = 'L160'
pv_power_forecast = PvPowerForecast(dwd_station_id)
```

It is essential that the PvPowerForecast library will be provided with real measured PV values of our PV system. 
The provided real data is used as train data to adapt the internal model on your environment. 
Providing additional technical parameters of your PV system such as installed power or cardinal direction is not required. This libray is self-learning. Technical parameters scuh as technical parameters of your PV system such as installed power or cardinal direction will be considered implicitly.
```
# please provide the real pv value periodically. The period should be between 1 minute and 15 minutes.
while True:
    real_pv_power_watt = ...read real PV power ...
    pv_power_forecast.add_train_sample(real_pv_power_watt)
    time.sleep(60)
```
The provided train data will be store internally on disc and be used to update the internal prediction model. Please consider, that more accurate forecast predictions require collecting real PV power data for at least 2 weeks. Do not stop providing real PV power data, even though the prediction becomes better and better. You may use a periodic job to provide the real PV values

To get the power forecast the power method has to be called 
```
tomorrow = datetime.now() + timedelta(days=1)
predicted_pv_power_watt_tomorrow = forecast.predict(tomorrow)
```

