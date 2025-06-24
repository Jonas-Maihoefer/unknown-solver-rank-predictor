set -x

# ensure USECUDA is defined (default to 0 if unset)
if [ -z "${USECUDA}" ]; then
  export USECUDA=0
fi

export PYTHONPATH="./src"
python src/al_experiments.py
