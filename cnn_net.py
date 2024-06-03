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

torch, nn = try_import_torch()

@OldAPIStack
class CustomNetwork(TorchModelV2, nn.Module):
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

        # Post FC net config.
        post_fcnet_hiddens = model_config.get("post_fcnet_hiddens", [])
        post_fcnet_activation = get_activation_fn(
            model_config.get("post_fcnet_activation"), framework="torch"
        )

        no_final_linear = self.model_config.get("no_final_linear")
        vf_share_layers = self.model_config.get("vf_share_layers")

        # Whether the last layer is the output of a Flattened (rather than
        # a n x (1,1) Conv2D).
        self.last_layer_is_flattened = False
        self._logits = None
        
        # Initial adjacent-summation filter
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
        # in_channels = 1

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
                    activation_fn=activation,
                )
            )
            in_channels = out_channels
            in_size = out_size

        layers.append(nn.Flatten())
        in_channels = np.prod(in_size) * out_channels

        if post_fcnet_hiddens:
            for size in post_fcnet_hiddens:
                layers.append(SlimFC(in_channels, size, activation_fn=post_fcnet_activation))
                in_channels = size

        self._logits = SlimFC(
            in_channels, num_outputs, activation_fn=None
        )

        self._convs = nn.Sequential(*layers)

        self._features = None

    @override(TorchModelV2)
    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> (TensorType, List[TensorType]):
        self._features = input_dict["obs"].float()
        self._features = self._features.unsqueeze(1)
        filter_out = torch.cat([f(self._features) for f in self.initial_filter], dim=1)
        conv_out = self._convs(filter_out)
        # conv_out = self._convs(self._features)

        if not self.last_layer_is_flattened:
            if self._logits:
                conv_out = self._logits(conv_out)
            if len(conv_out.shape) == 4:
                if conv_out.shape[2] != 1 or conv_out.shape[3] != 1:
                    raise ValueError(
                        "Given `conv_filters` ({}) do not result in a [B, {} "
                        "(`num_outputs`), 1, 1] shape (but in {})! Please "
                        "adjust your Conv2D stack such that the last 2 dims "
                        "are both 1.".format(
                            self.model_config["conv_filters"],
                            self.num_outputs,
                            list(conv_out.shape),
                        )
                    )
                logits = conv_out.squeeze(3)
                logits = logits.squeeze(2)
            else:
                logits = conv_out
            return logits, state
        else:
            return conv_out, state

    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        assert self._features is not None, "must call forward() first"
        # if self._value_branch_separate:
        #     value = self._value_branch_separate(self._features)
        #     value = value.squeeze(3)
        #     value = value.squeeze(2)
        #     return value.squeeze(1)
        # else:
        #     if not self.last_layer_is_flattened:
        #         features = self._features.squeeze(3)
        #         features = features.squeeze(2)
        #     else:
        #         features = self._features
        #     return self._value_branch(features).squeeze(1)
        return torch.sum((self._features == 0).int(), dim=(1, 2, 3)).float()

    def _hidden_layers(self, obs: TensorType) -> TensorType:
        res = self._convs(obs.permute(0, 3, 1, 2))  # switch to channel-major
        res = res.squeeze(3)
        res = res.squeeze(2)
        return res