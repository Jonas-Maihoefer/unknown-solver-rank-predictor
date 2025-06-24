# ensure USECUDA is defined (default to "0" if unset or empty)
if (-not $env:USECUDA) {
    $env:USECUDA = "0"
}

$env:PYTHONPATH="src"
python src/al_experiments.py
