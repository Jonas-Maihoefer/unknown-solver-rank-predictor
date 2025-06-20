import sys
import pandas as pd
import pickle
from main import start

with open(
        "../al-for-sat-solver-benchmarking-data/pickled-data/anni_full_df.pkl",
        "rb"
) as file:
    df: pd.DataFrame = pickle.load(file).copy()

USE_CUPY = True  # os.getenv("USE_CUPY") == "1"    # or set to "true", however you like

if USE_CUPY:
    print("start importing cupy")
    import cupy
    # Tell Python: whenever someone does `import numpy`, give them `cupy` instead
    sys.modules["numpy"] = cupy

start(df)
