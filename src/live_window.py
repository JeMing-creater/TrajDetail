"""Use the trained decoder's window format for a streaming test example."""
import numpy as np

from methods.trajdetail.train_shared_forecast import DOMAIN_CODES, Windows


class LiveWindow:
    pick = Windows.pick

    def __init__(self, axial, elapsed, interval, item, query, dataset):
        visible = item["context_times"][item["mask"] > 0]
        self.depth = axial.shape[-3]
        self.inputs = axial[None, None]
        self.targets = np.zeros((1, 1, *axial.shape[-3:]), dtype=np.float32)
        self.condition = [np.array([[elapsed, interval, len(visible) / 7., visible[-1],
                                    item["target_times"][query], *DOMAIN_CODES[dataset]]], dtype=np.float32)]
