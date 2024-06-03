import torch
import torch.nn as nn
from ray.rllib.policy.torch_policy import TorchPolicy
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.utils.annotations import override

class MaskedTorchPolicy(TorchPolicy):
    def __init__(self, observation_space, action_space, config):
        self.config = config
        self.model = config["model"]["custom_model"](observation_space, action_space, config)
        self.model.to(self.device)
        self.view_requirements = self.model.view_requirements

        super(MaskedTorchPolicy, self).__init__(
            observation_space,
            action_space,
            config,
            self.model,
            action_distribution_class=config["action_distribution_class"],
        )

    @override(TorchPolicy)
    def compute_actions(self, obs_batch, state_batches, prev_action_batch=None, prev_reward_batch=None, info_batch=None, episodes=None, **kwargs):
        action_mask = torch.tensor([info["action_mask"] for info in info_batch], device=self.device)
        logits, _ = self.model.forward({"obs": obs_batch})
        masked_logits = logits + (action_mask - 1) * 1e9  # 큰 음수를 더하여 불법적인 행동을 불가능하게 만듭니다.
        action_dist = self.dist_class(masked_logits, self.model)
        actions = action_dist.sample()

        return actions.cpu().numpy(), [], {}

# 정책 등록
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

class CustomModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        self.fc = nn.Linear(obs_space.shape[0], num_outputs)

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]
        logits = self.fc(obs)
        return logits, state

ModelCatalog.register_custom_model("my_model", CustomModel)