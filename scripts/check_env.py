import sys
import pandas as pd
import numpy as np
import yfinance as yf
import pyarrow
from dotenv import load_dotenv

print("Python:", sys.version.split()[0])
print("Executable:", sys.executable)
print("pandas", pd.__version__, "| numpy", np.__version__,
      "| yfinance", yf.__version__, "| pyarrow", pyarrow.__version__)

df = yf.download("GC=F", period="5d", interval="1d", progress=False, auto_adjust=False)
print("\nRows:", len(df))
print("Columns:", list(df.columns))
print("Index tz:", df.index.tz)
print(df.tail(3))