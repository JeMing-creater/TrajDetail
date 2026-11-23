"""Time-scaled extrapolation on the preprocessed intensity grid."""


def forecast(latest, earlier, ratio):
    return latest + (latest - earlier) * ratio
