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
    weatherForecast = json.loads(rawForecastData)
    df = pd.json_normalize(weatherForecast['days'][0])
    df.insert(loc=0, column='name', value='Olsztyn')
    df.rename(columns={'pressure': 'sealevelpressure'}, inplace=True)
    df.drop('datetimeEpoch', inplace=True, axis=1)
    df.drop('sunriseEpoch', inplace=True, axis=1)
    df.drop('sunsetEpoch', inplace=True, axis=1)
    df.drop('source', inplace=True, axis=1)
    return df


