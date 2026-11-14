class TimeSeriesValidationError(Exception):
    """Raised when time-series input has insufficient observations, NaNs, or invalid lags."""
    pass
