from methods.convlstm.model import ConvLSTMForecaster
from methods.node_lstm.model import NodeLSTMForecaster
from methods.simvp.model import SimVPForecaster
from methods.vivit.model import ViViTForecaster

MODEL_NAMES = ("convlstm", "simvp", "vivit", "node_lstm")

def build_forecaster(name: str, volume_shape=(16, 128, 128), width: int = 32, **kwargs):
    """Build one adapted predictor; unknown names fail instead of falling back."""
    if name == "convlstm":
        return ConvLSTMForecaster(volume_shape, width=width, **kwargs)
    if name == "simvp":
        return SimVPForecaster(volume_shape, width=width, **kwargs)
    if name == "vivit":
        return ViViTForecaster(volume_shape, width=width, **kwargs)
    if name == "node_lstm":
        return NodeLSTMForecaster(volume_shape, width=width, **kwargs)
    raise ValueError(f"unknown adapted baseline {name!r}; expected one of {MODEL_NAMES}")
