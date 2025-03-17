import numpy as np
import torch
import gymnasium as gym
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard import SummaryWriter
from env import BatteryBalancingEnv

# Custom Feature Extractor using CNN for local (1D-CFE) and Transformer for global (GDA) feature extraction.
class BatteryTransformerExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        self.num_cells = 10
        self.history_len = 5
        cell_feature_channels = 2
        derived_feature_dim = (self.num_cells - 1) * 4

        # 1D-CNN for local cell feature extraction
        self.cell_encoder = nn.Sequential(
            nn.Conv1d(
                in_channels=cell_feature_channels * self.history_len,
                out_channels=16,  # CNN Filters Layer 1
                kernel_size=1,    # Small kernel (stride 1)
                padding=0
            ),
            nn.LeakyReLU(),
            nn.LayerNorm([16, self.num_cells]),
            nn.Dropout(0.03),
            nn.Conv1d(
                in_channels=16,
                out_channels=16,  # CNN Filters Layer 2
                kernel_size=1,
                padding=0
            ),
            nn.LeakyReLU(),
            nn.LayerNorm([16, self.num_cells]),
            nn.Dropout(0.01)
        )
        # Transformer encoder (GDA) for global dependency modeling
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=16,       # Match CNN output channels
                nhead=8,          # Transformer Heads
                dim_feedforward=128,  # Feed-forward dimension
                dropout=0.1,
                activation=F.gelu,
                batch_first=True
            ),
            num_layers=3  # Number of Transformer blocks
        )
        # Processing derived features with current load appended later
        self.derived_feature_fc = nn.Sequential(
            nn.Linear(derived_feature_dim + 1, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(),
            nn.Dropout(0.2)
        )
        # Combining global and derived features into final global feature
        self.fc = nn.Sequential(
            nn.Linear(16 + 128, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, features_dim),
            nn.LayerNorm(features_dim),
            nn.LeakyReLU()
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]
        # Split observations into historical cell features and derived features with current load I(t)
        hist_cell_features = observations[:, :self.history_len * self.num_cells * 2].view(batch_size, self.num_cells, 2 * self.history_len)
        derived_features_current = observations[:, self.history_len * self.num_cells * 2: -1]
        current = observations[:, -1:]
        # Apply 1D-CNN for local feature extraction
        cnn_in = hist_cell_features.permute(0, 2, 1)
        cnn_out = self.cell_encoder(cnn_in)
        # Reshape for Transformer: each cell as a token for global attention
        transformer_in = cnn_out.permute(0, 2, 1)
        transformer_out = self.transformer(transformer_in)
        # Global pooling across cells to capture pack-level dynamics
        global_features = transformer_out.mean(dim=1)
        # Process derived features and current load I(t)
        derived_combined = torch.cat([derived_features_current, current], dim=1)
        derived_features_out = self.derived_feature_fc(derived_combined)
        # Final feature aggregation
        combined_features = torch.cat([global_features, derived_features_out], dim=1)
        return self.fc(combined_features)

# Custom PPO Policy using the CNN-Transformer extractor.
class BatteryPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            **kwargs,
            features_extractor_class=BatteryTransformerExtractor,
            net_arch=dict(pi=[128,64,32], vf=[128,64,32]),
            activation_fn=nn.ReLU,
            ortho_init=True
        )

    def _predict_vf(self, obs: torch.Tensor) -> torch.Tensor:
        return super()._predict_vf(obs)

    def _get_constructor_parameters(self) -> dict:
        data = super()._get_constructor_parameters()
        data.update(
            dict(
                net_arch=self.net_arch,
                activation_fn=self.activation_fn,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
            )
        )
        return data

    def _get_torch_save_params(self) -> tuple[list[str], list[str]]:
        state_dicts, modules = super()._get_torch_save_params()
        if self.features_extractor is not None:
            modules.append(self.features_extractor)
        return state_dicts, modules

# Callback for logging training metrics via TensorBoard.
class BatteryMonitor(BaseCallback):
    def __init__(self, verbose=0, log_freq=300):
        super().__init__(verbose)
        self.writer = None
        self.step_count = 0
        self.log_freq = log_freq

    def _on_training_start(self) -> None:
        log_dir = self.model.tensorboard_log if hasattr(self.model, "tensorboard_log") else "./logs/"
        self.writer = SummaryWriter(log_dir)

    def _on_step(self) -> bool:
        if self.step_count % self.log_freq == 0:
            if "infos" in self.locals:
                for info in self.locals["infos"]:
                    if "voltage_std" in info:
                        self._log_metrics(info)
        self.step_count += 1
        return True

    def _log_metrics(self, info):
        for key in ["soc_std", "voltage_std", "soc_diff_abs_sum", "v_diff_abs_sum"]:
            value = info.get(key, 0)
            self.writer.add_scalar(f"metrics/{key}", value, self.step_count)
        for i, diff in enumerate(info.get("soc_diffs", [])):
            try:
                self.writer.add_scalar(f"metrics/soc_diff_{i}-{i+1}", float(diff), self.step_count)
            except ValueError:
                print(f"Warning: Could not convert soc_diff[{i}] to float: {diff}")
        for i, diff in enumerate(info.get("v_diffs", [])):
            try:
                self.writer.add_scalar(f"metrics/v_diff_{i}-{i+1}", float(diff), self.step_count)
            except ValueError:
                print(f"Warning: Could not convert v_diff[{i}] to float: {diff}")
        self.writer.flush()

# Callback for periodic PPO model checkpoint saving.
class SaveModelCallbackPPO(BaseCallback):
    def __init__(self, save_interval, save_path, verbose=0):
        super().__init__(verbose)
        self.save_interval = save_interval
        self.save_path = save_path
        self.step_count = 0

    def _on_step(self) -> bool:
        self.step_count += self.training_env.num_envs
        if self.step_count >= self.save_interval:
            self.step_count %= self.save_interval
            path = f"{self.save_path}_{self.num_timesteps // 1000000}M"
            self.model.save(path)
            if self.verbose > 1:
                print(f"Saving PPO model checkpoint at {self.num_timesteps} steps to {path}")
        return True

# Training function using PPO with the CNN-Transformer feature extractor.
def train():
    # Create vectorized environment
    vec_env = make_vec_env(lambda: BatteryBalancingEnv(), n_envs=60)
    n_steps_per_rollout = 512 * 4  # Rollout steps per update

    # PPO hyperparameters (aligned with proposed values)
    batch_size_ppo = 256         # Minibatch Size
    clip_range_ppo = 0.1         # Clipping Threshold
    target_kl_ppo = 0.05         # Target KL divergence
    ent_coef_ppo = 0.002         # Entropy Coefficient
    vf_coef_ppo = 1.0            # Value Loss Coefficient
    gae_lambda_ppo = 0.97        # GAE Parameter
    n_epochs_ppo = 20            # PPO Epochs per Update

    # Initialize PPO model with the custom policy
    model = PPO(
        BatteryPolicy,
        vec_env,
        learning_rate=3.5e-4,    # Learning Rate
        n_steps=n_steps_per_rollout,
        batch_size=batch_size_ppo,
        gamma=0.9999,           # Discount Factor
        gae_lambda=gae_lambda_ppo,
        clip_range=clip_range_ppo,
        target_kl=target_kl_ppo,
        ent_coef=ent_coef_ppo,
        vf_coef=vf_coef_ppo,
        n_epochs=n_epochs_ppo,
        max_grad_norm=0.5,      # Gradient Clipping Norm
        verbose=1,
        tensorboard_log="./Test_01/logs/",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    # Apply L2 regularization (weight decay)
    weight_decay_value = 1e-5
    optimizer = model.policy.optimizer
    for param_group in optimizer.param_groups:
        param_group['weight_decay'] = weight_decay_value
    # Define callbacks for model monitoring and saving
    save_callback = SaveModelCallbackPPO(save_interval=1000000, save_path="./Test_01/models/cnn_transformer", verbose=1)
    # Start PPO training
    model.learn(
        total_timesteps=20_000_000,
        callback=[BatteryMonitor(log_freq=1), save_callback],
        progress_bar=True
    )
    # Save the final trained model
    model.save("./Test_01/cnn_transformer")

if __name__ == "__main__":
    train()
