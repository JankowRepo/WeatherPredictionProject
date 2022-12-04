import pandas as pd
import numpy as np
import psycopg2
from sqlalchemy import create_engine

def read_data(DB_path):
    engine = create_engine(DB_path)
    connection = engine.connect()
    df = pd.read_sql("""SELECT * FROM public."DBWeather" """, con=connection)
    return df