import pandas as pd
import numpy as np
import DB_reader as db
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
import matplotlib.ticker as mtick
import importlib

importlib.reload(db)

def get_data():
    # df=db.read_data('postgresql+psycopg2://postgres:admin@localhost/postgres')
    df = db.read_data('postgresql+psycopg2://postgres:adam123@localhost/postgres')

    df = df[["datetime", "temp", "tempmin", "tempmax", "feelslike", "feelslikemax", "feelslikemin",
             'dew', "humidity", "precip", 'preciptype', 'precipcover', 'snowdepth', "windspeed",
             'winddir', 'sealevelpressure', 'cloudcover', 'visibility', 'moonphase', 'conditions']]

    df.loc[df.preciptype == '', 'preciptype'] = 0
    df.loc[df.preciptype == 'rain', 'preciptype'] = 1
    df.loc[df.preciptype == 'snow', 'preciptype'] = 2
    df.loc[df.preciptype == 'rain,snow', 'preciptype'] = 3
    df.loc[df.preciptype == 'rain,freezingrain', 'preciptype'] = 4
    df["preciptype"] = pd.to_numeric(df["preciptype"])


    df.snowdepth = df.snowdepth.fillna(0)


    df.loc[df.conditions == 'Clear', 'conditions'] = 0
    df.loc[df.conditions == 'Partially cloudy', 'conditions'] = 0
    df.loc[df.conditions == 'Overcast', 'conditions'] = 0

    df.loc[df.conditions == 'Rain', 'conditions'] = 1
    df.loc[df.conditions == 'Rain, Overcast', 'conditions'] = 1
    df.loc[df.conditions == 'Rain, Fog', 'conditions'] = 1
    df.loc[df.conditions == 'Rain, Partially cloudy', 'conditions'] = 1

    df.loc[df.conditions == 'Snow, Rain', 'conditions'] = 2
    df.loc[df.conditions == 'Snow, Rain, Overcast', 'conditions'] = 2
    df.loc[df.conditions == 'Snow, Rain, Fog', 'conditions'] = 2
    df.loc[df.conditions == 'Snow, Rain, Partially cloudy', 'conditions'] = 2
    df.loc[df.conditions == 'Rain, Freezing Drizzle/Freezing Rain, Overcast', 'conditions'] = 2

    df.loc[df.conditions == 'Snow', 'conditions'] = 3
    df.loc[df.conditions == 'Snow, Overcast', 'conditions'] = 3
    df.loc[df.conditions == 'Snow, Fog', 'conditions'] = 3
    df.loc[df.conditions == 'Snow, Partially cloudy', 'conditions'] = 3


    df["conditions"] = pd.to_numeric(df["conditions"])


    df['datetime'] = pd.to_datetime(df['datetime'])
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    df['day_of_year'] = df['datetime'].dt.dayofyear
    df = df.drop(columns=['datetime'], axis=1)

    df['nextday_rainfall'] = np.where(df['preciptype'] > 0, 1, 0)
    df['nextday_rainfall'] = df['nextday_rainfall'].shift(-1)
    df.dropna()

    scaler = MinMaxScaler()
    df = pd.DataFrame(scaler.fit_transform(df), index=df.index, columns=df.columns)

    best_features = np.abs(df.corr()['nextday_rainfall']).sort_values(ascending=False) > 0.05
    best_features = best_features.where(best_features.values == True).dropna().index
    df = df[best_features]

    return df
