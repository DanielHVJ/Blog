import pandas as pd

rus = pd.read_csv("E:/GitRepo/Blog/rusia_rides.csv", sep=";")

rus.drop("trip_completed_at", inplace=True, axis=1)
rus.head(10)

## Trip end time

rus["date"] = [d for d in rus["trip_end_time"]]
rus["time"] = [d for d in rus["trip_end_time"]]

# rus['trip_end_time'] = pd.to_datetime(rus['trip_end_time'])

rus.drop("trip_end_time", inplace=True, axis=1)

## Trip start time
# rus['trip_time'] = pd.Series([val.time() for val in rus['trip_time']])

rus["s_date"] = [d for d in rus["trip_time"]]
rus["s_time"] = [d for d in rus["trip_time"]]

## Converting time

time = pd.DatetimeIndex(rus["total_time"])
rus["total_time"] = time.hour * 60 + time.minute

wait = pd.DatetimeIndex(rus["wait_time"])
rus["wait_time"] = wait.hour * 60 + wait.minute

# rus['total_time'] = pd.to_datetime(rus['total_time'])
# rus['wait_time'] = pd.to_datetime(rus['wait_time'])

# rus['total_time'] = [d.time() for d in rus['total_time']]
# rus['wait_time'] = [d.time() for d in rus['wait_time']]

## Changing the values for gender variable

rus["driver_gender"] = rus["driver_gender"].replace("Male", 1)
rus["driver_gender"] = rus["driver_gender"].replace("Female", 0)

data = rus[
    [
        "trip_status",
        "ride_hailing_app",
        "trip_time",
        "total_time",
        "wait_time",
        "trip_type",
        "surge_multiplier",
        "vehicle_make",
        "driver_gender",
        "trip_map_image_url",
        "price_usd",
        "distance_kms",
        "temperature_value",
        "humidity",
        "wind_speed",
        "cloudness",
        "weather_main",
        "weather_desc",
        "precipitation",
    ]
]

## save file ----------------------------
data.to_csv(r"E:/GitRepo/Blog/rus2.csv", index=None, header=True)

data.head()
data.shape
