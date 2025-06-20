import sys

USE_CUPY = True  # os.getenv("USE_CUPY") == "1"    # or set to "true", however you like

if USE_CUPY:
    import cupy
    # Tell Python: whenever someone does `import numpy`, give them `cupy` instead
    sys.modules["numpy"] = cupy

# now import your real entry point
import al_experiments
