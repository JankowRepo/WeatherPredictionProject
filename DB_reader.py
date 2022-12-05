import urllib.parse
import urllib.request
import json
import pandas as pd
import numpy as np
import psycopg2
from sqlalchemy import create_engine

def read_data(DB_path):
    engine = create_engine(DB_path)
    connection = engine.connect()
    df = pd.read_sql("""SELECT * FROM public."DBWeather" """, con=connection)
    return df


API_KEY = "D83SDPLMVBRWQMWP5QAVLQ9G4"
LOCATION = "Olsztyn,PL"
UNIT_GROUP = "metric"


def getWeatherForecast():
    requestUrl = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/" + urllib.parse.quote_plus(
        LOCATION)
    requestUrl = requestUrl + "?key=" + API_KEY + "&unitGroup=" + UNIT_GROUP + "&include=days";

    print('Weather requestUrl={requestUrl}'.format(requestUrl=requestUrl))

    try:
        req = urllib.request.urlopen(requestUrl)
    except:
        print("Could not read from:" + requestUrl);
        return []

    rawForecastData = req.read()
    req.close()
    return json.loads(rawForecastData)

weatherForecast = getWeatherForecast()
print('Weather forecast for {location}'.format(location=weatherForecast['resolvedAddress']))
days=weatherForecast['days'];

for day in days:
    print('{datetime} tempmax:{tempmax} tempmin:{tempmin} description:{description}'.format(datetime=day['datetime'], tempmax=day["tempmax"], tempmin=day["tempmin"], description=day["description"]))