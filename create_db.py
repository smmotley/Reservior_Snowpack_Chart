import sqlite3
import os
import time
import datetime
import pandas as pd

DEFAULT_PATH = os.path.join(os.path.dirname(__file__), 'test_db.sqlite3')

def db_connect(db_path=DEFAULT_PATH):
    con = sqlite3.connect(db_path)
    return con

def create_table(c):
    #c.execute("CREATE TABLE IF NOT EXISTS stuffToPlot(unix REAL, datestamp TEXT, keyword TEXT, value REAL)")
    #c.execute("CREATE TABLE IF NOT EXISTS cities(city_id TEXT, city_name TEXT, lat REAL, lon REAL)")
    c.execute("CREATE TABLE IF NOT EXISTS basin_swe("
                "date_valid TIMESTAMP , "
                "basin TEXT, "
                "Max_Temp REAL, "
                "Min_Temp REAL, "
                "precip REAL,"
                "FOREIGN KEY (basin) REFERENCES cities(city_id))")
    c.execute("CREATE TABLE IF NOT EXISTS forecasts("
                "city_code TEXT, "
                "date_created TIMESTAMP , "
                "date_valid TIMESTAMP , "
                "forecast_day REAL, "
                "GFS REAL, "
                "GFS_bc REAL,"
                "EURO REAL, "
                "EURO_EPS REAL, "
                "NWS REAL, "
                "PCWA REAL )"
              )

df = pd.read_excel("Daily_Output.xlsx", sheet_name="French_Meadows")
conn = db_connect()
c = conn.cursor()
#del_and_update()
#graph_data()
#read_from_db()
create_table(c)
#data_entry(c,conn)
# for i in range(10):
#     dynamic_data_entry()
#     time.sleep(1)
c.close()
conn.close()