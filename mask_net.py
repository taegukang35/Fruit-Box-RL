import numpy as np
from typing import Dict, List
import gymnasium as gym

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import (
    normc_initializer,
    same_padding,
    SlimConv2d,
    SlimFC,
)
from ray.rllib.models.utils import get_activation_fn, get_filter_config
from ray.rllib.utils.annotations import OldAPIStack, override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import ModelConfigDict, TensorType

import torch
from torch import nn
import torch.nn.functional as F

class ArgMaxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        inp = input.flatten(start_dim=-2)
        flat_idx = torch.argmax(inp, dim=-1)
        i = flat_idx // input.shape[-1]
        j = flat_idx % input.shape[-1]
        return torch.cat([i, j], dim=-1)

    @staticmethod
    def backward(ctx, grad_output):
        return F.hardtanh(grad_output)

class StraightThroughEstimator(nn.Module):
    def __init__(self):
        super(StraightThroughEstimator, self).__init__()

    def forward(self, x):
            x = ArgMaxFunction.apply(x)
            return x

@OldAPIStack
class MaskNetwork(TorchModelV2, nn.Module):
    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
    ):
        if not model_config.get("conv_filters"):
            model_config["conv_filters"] = get_filter_config(obs_space.shape)

        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        activation = self.model_config.get("conv_activation")
        filters = self.model_config["conv_filters"]
        assert len(filters) > 0, "Must provide at least 1 entry in `conv_filters`!"

        vf_share_layers = self.model_config.get("vf_share_layers")

        self.initial_filter = []
        init_kernels = [(1, 2), (2, 1)]
        for kernel in init_kernels:
            init_filter = nn.Conv2d(1, 1, kernel, stride=1, padding='same', bias=False)
            init_filter.weight.data = torch.ones_like(init_filter.weight.data)
            init_filter.weight.requires_grad = False
            self.initial_filter.append(init_filter)

        layers = []
        (w, h) = obs_space.shape
        in_channels = 2

        in_size = [w, h]
        for out_channels, kernel, stride in filters:
            padding, out_size = same_padding(in_size, kernel, stride)
            layers.append(
                SlimConv2d(
                    in_channels,
                    out_channels,
                    kernel,
                    stride,
                    padding,
                    activation_fn=activation
                )
            )
            in_channels = out_channels
            in_size = out_size

        layers.append(
            SlimConv2d(
                in_channels,
                2,
                1,
                1,
                None,
                activation_fn=nn.Sigmoid,
            )
        )

        self._convs = nn.Sequential(*layers)
        self._final = SlimFC(
            in_size=2,
        )

        # Build the value layers
        self._value_branch_separate = self._value_branch = None
        if vf_share_layers:
            self._value_branch = SlimFC(
                out_channels, 1, initializer=normc_initializer(0.01), activation_fn=None
            )
        else:
            vf_layers = []
            (w, h) = obs_space.shape
            in_channels = 1
            in_size = [w, h]
            for out_channels, kernel, stride in filters[:-1]:
                padding, out_size = same_padding(in_size, kernel, stride)
                vf_layers.append(
                    SlimConv2d(
                        in_channels,
                        out_channels,
                        kernel,
                        stride,
                        padding,
                        activation_fn=activation,
                    )
                )
                in_channels = out_channels
                in_size = out_size

            out_channels, kernel, stride = filters[-1]
            vf_layers.append(
                SlimConv2d(
                    in_channels,
                    out_channels,
                    kernel,
                    stride,
                    None,
                    activation_fn=activation,
                )
            )

            vf_layers.append(
                SlimConv2d(
                    in_channels=out_channels,
                    out_channels=1,
                    kernel=1,
                    stride=1,
                    padding=None,
                    activation_fn=None,
                )
            )
            self._value_branch_separate = nn.Sequential(*vf_layers)

        # Holds the current "base" output (before logits layer).
        self._features = None

    @override(TorchModelV2)
    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> (TensorType, List[TensorType]):
        self._features = input_dict["obs"].float()
        # Permuate b/c data comes in as [B, dim, dim, channels]:
        # self._features = self._features.permute(0, 3, 1, 2)
        self._features = self._features.unsqueeze(1)
        filter_out = torch.cat([f(self._features) for f in self.initial_filter], dim=1)
        conv_out = self._convs(filter_out)
        print('KONTOL')
        print(conv_out.shape)
        choice = self._argmax(conv_out)
        print(choice.shape)
        out = torch.zeros((self._features.shape[0], self._features.shape[-2], self._features.shape[-1], self._features.shape[-2], self._features.shape[-1]))
        # Store features to save forward pass when getting value_function out.
        if not self._value_branch_separate:
            self._features = conv_out
        
        return choice, state

    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        assert self._features is not None, "must call forward() first"
        return torch.sum((self._features == 0).int(), dim=(1, 2, 3)).float()
    
    def _hidden_layers(self, obs: TensorType) -> TensorType:
        res = self._convs(obs.permute(0, 3, 1, 2))  # switch to channel-major
        res = res.squeeze(3)
        res = res.squeeze(2)
        return res