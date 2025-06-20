import sys
from main import start

USE_CUPY = True  # os.getenv("USE_CUPY") == "1"    # or set to "true", however you like

if USE_CUPY:
    print("start importing cupy")
    import cupy
    # Tell Python: whenever someone does `import numpy`, give them `cupy` instead
    sys.modules["numpy"] = cupy

start()
