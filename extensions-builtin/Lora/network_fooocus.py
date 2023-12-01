import torch

import lyco_helpers
import network
from modules import devices


class ModuleTypeFooocus(network.ModuleType):
    def create_module(self, net: network.Network, weights: network.NetworkWeights):
        if all(x in weights.w for x in ["w.weight", "w_min.weight", "w_max.weight"]):
            return NetworkModuleFooocus(net, weights)

        return None


class NetworkModuleFooocus(network.NetworkModule):
    def __init__(self,  net: network.Network, weights: network.NetworkWeights):
        super().__init__(net, weights)
        self.weight = (weights.w["w.weight"], weights.w["w_min.weight"], weights.w["w_max.weight"])
        self.ex_bias = (weights.w.get("w.bias"), weights.w.get("w_min.bias"), weights.w.get("w_max.bias")) if "w.bias" in weights.w else None

    def calc_updown(self, orig_weight):
        def calculate_weight(w1, w_min, w_max):
            w1 = w1.to(orig_weight.device, dtype=orig_weight.dtype)
            w_min = w_min.to(orig_weight.device, dtype=orig_weight.dtype)
            w_max = w_max.to(orig_weight.device, dtype=orig_weight.dtype)
            return (w1 / 255.0) * (w_max - w_min) + w_min

        updown = calculate_weight(*self.weight)
        ex_bias = calculate_weight(*self.ex_bias) if self.ex_bias is not None else None
        output_shape = orig_weight.shape

        return self.finalize_updown(updown, orig_weight, output_shape, ex_bias=ex_bias)
