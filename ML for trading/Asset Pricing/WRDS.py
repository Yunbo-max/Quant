import pandas as pd
import numpy as np
import sqlite3

from plotnine import *
from mizani.formatters import comma_format, percent_format
from datetime import datetime

start_date = "01/01/1960"
end_date = "12/31/2015"

import os
from dotenv import load_dotenv
load_dotenv()
from sqlalchemy import create_engine

connection_string = (
  "postgresql+psycopg2://"
 f"{os.getenv('yunbo')}:{os.getenv('longfu118879')}"
  "@wrds-pgdata.wharton.upenn.edu:9737/wrds"
)

wrds = create_engine(connection_string, pool_pre_ping=True)


crsp_monthly_query = (
  "SELECT msf.permno, msf.date, "
         "date_trunc('month', msf.date)::date as month, "
         "msf.ret, msf.shrout, msf.altprc, "
         "msenames.exchcd, msenames.siccd, "
         "msedelist.dlret, msedelist.dlstcd "
  
   f"WHERE msf.date BETWEEN '{start_date}' AND '{end_date}' "
          "AND msenames.shrcd IN (10, 11)"
)

crsp_monthly = (pd.read_sql_query(
    sql=crsp_monthly_query,
    con=wrds,
    dtype={"permno": int, "exchcd": int, "siccd": int},
    parse_dates={"date", "month"})
  .assign(shrout=lambda x: x["shrout"]*1000)
)
