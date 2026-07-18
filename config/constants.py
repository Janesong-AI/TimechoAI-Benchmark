"""
config/constants.py -- Global Constants Definition
Defines global constants, including default models, default parameters, etc.
"""

# ============================================================
# Default Model Parameters
# ============================================================
DEFAULT_MODEL_ID = "Timer-3.5"
MODEL_LIST = [
    "Auto",          # Automatic selection
    "Timer-3.5",     # Timecho flagship model
    "Timer-3.0",     # Previous generation
    "Chronos-2",     # Developed by Amazon
    "AutoARIMA",     # Statistical method, AutoRegressive Integrated Moving Average
    "Holt-Winters",  # Statistical method, Holt-Winters model / Triple exponential smoothing
]

# ============================================================
# Default Forecasting Parameters
# ============================================================
DEFAULT_INPUT_LENGTH = 256    # Input 256 historical points
DEFAULT_OUTPUT_LENGTH = 64    # Predict 64 future points

