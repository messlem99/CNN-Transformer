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
from torch.utils.tensorboard import SummaryWriter
from env import BatteryBalancingEnv
from optuna.trial import Trial
from optuna.pruners import MedianPruner
from tensorboard.plugins.hparams import api as hp
import tensorflow as tf
from typing import Dict
import shutil
import optuna
import os

# Disable OneDNN optimizations in TF for consistency
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
print(f"PyTorch version: {torch.__version__}")
print("SB3 version: (run in terminal 'pip show stable-baselines3')")

# Global directories and constants for logging and study management
OPTUNA_FOLDER = "Optuna/"
BASE_NAME = "battery_optuna"
SB3_LOG_DIR = os.path.join(OPTUNA_FOLDER, "sb3_logs/")
HPARAMS_LOG_DIR = os.path.join(OPTUNA_FOLDER, "hparams_logs/")
STUDY_NAME = "battery_hpo"
DATABASE_FILE = f"sqlite:///{OPTUNA_FOLDER}{BASE_NAME}.db"
METRIC_NAME = "soc_diff_abs_sum"

# Define HParams for TensorBoard HParams Dashboard
HP_CNN_KERNEL_SIZE = hp.HParam('cnn_kernel_size', hp.Discrete([1, 3, 5]))
HP_CNN_FILTERS_LAYER1 = hp.HParam('cnn_filters_layer1', hp.Discrete([16, 32, 64]))
HP_CNN_FILTERS_LAYER2 = hp.HParam('cnn_filters_layer2', hp.Discrete([16, 32, 64, 128]))
HP_TRANSFORMER_BLOCKS = hp.HParam('transformer_blocks', hp.Discrete([1, 2, 3]))
HP_TRANSFORMER_HEADS = hp.HParam('transformer_heads', hp.Discrete([4, 8, 12]))
HP_TRANSFORMER_FF_DIM = hp.HParam('transformer_ff_dim', hp.Discrete([128, 256, 512]))

HP_LR = hp.HParam('lr', hp.RealInterval(1e-5, 1e-3))
HP_CLIP_RANGE = hp.HParam('clip_range', hp.Discrete([0.1, 0.15, 0.2]))
HP_GAE_LAMBDA = hp.HParam('gae_lambda', hp.Discrete([0.90, 0.95, 0.97]))
HP_N_EPOCHS = hp.HParam('n_epochs', hp.Discrete([5, 10, 15]))
HP_ENT_COEF = hp.HParam('ent_coef', hp.Discrete([0.001, 0.01]))
HP_VF_COEF = hp.HParam('vf_coef', hp.Discrete([0.5, 1.0]))
HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([64, 128, 256]))
HP_N_STEPS = hp.HParam('n_steps', hp.Discrete([2048, 8192]))

# ---------------------------
# Custom Feature Extractor
# ---------------------------
class BatteryTransformerExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 512,
                 cnn_kernel_size=1, cnn_filters_layer1=16, cnn_filters_layer2=16,
                 transformer_blocks=3, transformer_heads=8, transformer_ff_dim=128):
        super().__init__(observation_space, features_dim)
        self.num_cells = 10
        self.history_len = 5
        cell_feature_channels = 2
        derived_feature_dim = (self.num_cells - 1) * 4

        # Adjust padding so the sequence length is preserved
        padding = cnn_kernel_size // 2

        # 1D-CNN for local feature extraction
        self.cell_encoder = nn.Sequential(
            nn.Conv1d(in_channels=cell_feature_channels * self.history_len,
                      out_channels=cnn_filters_layer1,
                      kernel_size=cnn_kernel_size,
                      padding=padding),
            nn.LeakyReLU(),
            nn.LayerNorm([cnn_filters_layer1, self.num_cells]),
            nn.Dropout(0.03),
            nn.Conv1d(in_channels=cnn_filters_layer1,
                      out_channels=cnn_filters_layer2,
                      kernel_size=cnn_kernel_size,
                      padding=padding),
            nn.LeakyReLU(),
            nn.LayerNorm([cnn_filters_layer2, self.num_cells]),
            nn.Dropout(0.01)
        )

        # Transformer encoder for global dependency modeling
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=cnn_filters_layer2,
                nhead=transformer_heads,
                dim_feedforward=transformer_ff_dim,
                dropout=0.1,
                activation=F.gelu,
                batch_first=True
            ),
            num_layers=transformer_blocks
        )

        # FC layer for derived features (with current load appended later)
        self.derived_feature_fc = nn.Sequential(
            nn.Linear(derived_feature_dim + 1, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(),
            nn.Dropout(0.2)
        )
        # Final feature aggregation to produce global feature
        self.fc = nn.Sequential(
            nn.Linear(cnn_filters_layer2 + 128, 512),
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
        hist_cell_features = observations[:, :self.history_len * self.num_cells * 2].view(
            batch_size, self.num_cells, 2 * self.history_len)
        derived_features_current = observations[:, self.history_len * self.num_cells * 2: -1]
        current = observations[:, -1:]
        # Apply 1D-CNN for local features
        cnn_in = hist_cell_features.permute(0, 2, 1)
        cnn_out = self.cell_encoder(cnn_in)
        # Reshape for Transformer (each cell as a token)
        transformer_in = cnn_out.permute(0, 2, 1)
        transformer_out = self.transformer(transformer_in)
        # Global pooling over cells to capture pack-level context
        global_features = transformer_out.mean(dim=1)
        # Process derived features and current load
        derived_combined = torch.cat([derived_features_current, current], dim=1)
        derived_features_out = self.derived_feature_fc(derived_combined)
        # Final feature aggregation
        combined_features = torch.cat([global_features, derived_features_out], dim=1)
        return self.fc(combined_features)

# ---------------------------
# Custom PPO Policy
# ---------------------------
class BatteryPolicy(ActorCriticPolicy):
    def __init__(self, *args, features_extractor_kwargs: Dict = None, **kwargs):
        super().__init__(
            *args,
            **kwargs,
            features_extractor_class=BatteryTransformerExtractor,
            features_extractor_kwargs=features_extractor_kwargs,
            net_arch=dict(pi=[128, 64], vf=[128, 64]),
            activation_fn=nn.ReLU,
            ortho_init=True,
        )

    def _predict_vf(self, obs: torch.Tensor) -> torch.Tensor:
        return super()._predict_vf(obs)

    def _get_constructor_parameters(self) -> dict:
        data = super()._get_constructor_parameters()
        data.update(dict(
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            features_extractor_class=self.features_extractor_class,
            features_extractor_kwargs=self.features_extractor_kwargs,
        ))
        return data

    def _get_torch_save_params(self) -> tuple[list[str], list[nn.Module]]:
        state_dicts, modules = super()._get_torch_save_params()
        if self.features_extractor is not None:
            modules.append(self.features_extractor)
        return state_dicts, modules

# ---------------------------
# Monitoring Callback for Optuna Trials
# ---------------------------
class BatteryMonitor(BaseCallback):
    def __init__(self, trial: optuna.trial.Trial, verbose=0, log_freq=300):
        super().__init__(verbose)
        self.trial = trial
        self.writer = None
        self.step_count = 0
        self.log_freq = log_freq
        self.trial_step = 0
        self.metrics_dict = {}
        self.best_metric = float('inf')

    def _on_training_start(self) -> None:
        log_dir = os.path.join(SB3_LOG_DIR, f"trial_{self.trial.number}")
        self.writer = SummaryWriter(log_dir)

    def _on_step(self) -> bool:
        if self.step_count % self.log_freq == 0:
            if "infos" in self.locals:
                for info in self.locals["infos"]:
                    if "soc_diff_abs_sum" in info:
                        metrics = self._log_metrics(info)
                        soc_diff = metrics["soc_diff_abs_sum"]
                        self.trial.report(soc_diff, self.trial_step)

                        if soc_diff < self.best_metric:
                            self.best_metric = soc_diff
                            save_path = os.path.join(SB3_LOG_DIR, f"best_model_trial_{self.trial.number}")
                            self.model.save(save_path)

                        if self.trial.should_prune():
                            raise optuna.exceptions.TrialPruned()
                        self.trial_step += 1
        self.step_count += 1
        return True

    def _log_metrics(self, info):
        metrics = {
            "voltage_std": info.get("voltage_std", 0),
            "soc_std": info.get("soc_std", 0),
            "soc_diff_abs_sum": info.get("soc_diff_abs_sum", 0),
            "v_diff_abs_sum": info.get("v_diff_abs_sum", 0)
        }
        if "actions" in self.locals and self.locals["actions"] is not None:
            metrics["action_std"] = np.std(self.locals["actions"])
        else:
            metrics["action_std"] = 0

        self.metrics_dict = metrics

        for key, value in metrics.items():
            self.writer.add_scalar(f"metrics/{key}", value, self.step_count)

        adjacent_soc_diffs = info.get("soc_diffs")
        if adjacent_soc_diffs is not None:
            for i, diff in enumerate(adjacent_soc_diffs):
                try:
                    self.writer.add_scalar(f"metrics/soc_diff_{i}-{i+1}", float(diff), self.step_count)
                except ValueError:
                    print(f"Warning: Could not convert soc_diff[{i}] to float: {diff}")

        adjacent_v_diffs = info.get("v_diffs")
        if adjacent_v_diffs is not None:
            for i, diff in enumerate(adjacent_v_diffs):
                try:
                    self.writer.add_scalar(f"metrics/v_diff_{i}-{i+1}", float(diff), self.step_count)
                except ValueError:
                    print(f"Warning: Could not convert v_diff[{i}] to float: {diff}")
        self.writer.flush()
        return self.metrics_dict

# ---------------------------
# Objective Function for Optuna
# ---------------------------
def objective(trial: optuna.trial.Trial) -> float:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Suggest hyperparameters for CNN-Transformer architecture and PPO algorithm.
    cnn_kernel_size = trial.suggest_categorical('cnn_kernel_size', [1, 3, 5])
    cnn_filters_layer1 = trial.suggest_categorical('cnn_filters_layer1', [16, 32, 64])
    cnn_filters_l2 = trial.suggest_categorical('cnn_filters_layer2', [16, 32, 64, 128])
    valid_transformer_heads = [h for h in [4, 8, 12] if cnn_filters_l2 % h == 0]
    if not valid_transformer_heads:
        valid_transformer_heads = [4]
    transformer_heads = trial.suggest_categorical('transformer_heads', valid_transformer_heads)
    transformer_blocks = trial.suggest_categorical('transformer_blocks', [1, 2, 3])
    transformer_ff_dim = trial.suggest_categorical('transformer_ff_dim', [128, 256, 512])
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    clip_range = trial.suggest_categorical('clip_range', [0.1, 0.15, 0.2])
    gae_lambda = trial.suggest_categorical('gae_lambda', [0.90, 0.95, 0.97])
    n_epochs = trial.suggest_categorical('n_epochs', [5, 10, 15])
    ent_coef = trial.suggest_categorical('ent_coef', [0.001, 0.01])
    vf_coef = trial.suggest_categorical('vf_coef', [0.5, 1.0])
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256])
    n_steps = trial.suggest_categorical('n_steps', [2048, 8192])

    hparams = {
        HP_CNN_KERNEL_SIZE: cnn_kernel_size,
        HP_CNN_FILTERS_LAYER1: cnn_filters_layer1,
        HP_CNN_FILTERS_LAYER2: cnn_filters_l2,
        HP_TRANSFORMER_BLOCKS: transformer_blocks,
        HP_TRANSFORMER_HEADS: transformer_heads,
        HP_TRANSFORMER_FF_DIM: transformer_ff_dim,
        HP_LR: lr,
        HP_CLIP_RANGE: clip_range,
        HP_GAE_LAMBDA: gae_lambda,
        HP_N_EPOCHS: n_epochs,
        HP_ENT_COEF: ent_coef,
        HP_VF_COEF: vf_coef,
        HP_BATCH_SIZE: batch_size,
        HP_N_STEPS: n_steps,
    }

    feature_extractor_kwargs = {
        'cnn_kernel_size': cnn_kernel_size,
        'cnn_filters_layer1': cnn_filters_layer1,
        'cnn_filters_layer2': cnn_filters_l2,
        'transformer_blocks': transformer_blocks,
        'transformer_heads': transformer_heads,
        'transformer_ff_dim': transformer_ff_dim,
    }

    # Create vectorized environment with 70 parallel instances.
    vec_env = make_vec_env(lambda: BatteryBalancingEnv(), n_envs=70)

    # Initialize PPO model using the custom BatteryPolicy with CNN-Transformer extractor.
    model = PPO(
        BatteryPolicy,
        vec_env,
        learning_rate=lr,
        n_steps=n_steps,
        batch_size=batch_size,
        gamma=0.9999,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        target_kl=0.05,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        n_epochs=n_epochs,
        max_grad_norm=0.5,
        verbose=0,
        tensorboard_log=SB3_LOG_DIR,
        device=device,
        policy_kwargs=dict(features_extractor_kwargs=feature_extractor_kwargs)
    )

    print(f"Model device after creation: {next(model.policy.parameters()).device}")

    monitor_callback = BatteryMonitor(trial=trial, log_freq=5)
    monitor_callback.model = model

    total_timesteps_trial = 500000 
    model.learn(
        total_timesteps=total_timesteps_trial,
        callback=[monitor_callback],
        progress_bar=False
    )

    mean_reward, std_reward = evaluate_policy(model.policy, vec_env, n_eval_episodes=5, warn=False)
    final_soc_diff = monitor_callback.metrics_dict.get("soc_diff_abs_sum", float('inf'))
    vec_env.close()

    # Log hyperparameters and the objective metric to TensorBoard
    run_name = f"trial_{trial.number}"
    run_dir = os.path.join(HPARAMS_LOG_DIR, run_name)
    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams)
        tf.summary.scalar(METRIC_NAME, final_soc_diff, step=1)
    return final_soc_diff

if __name__ == "__main__":
    os.makedirs(OPTUNA_FOLDER, exist_ok=True)

    # Remove previous logs and database file for a clean start
    for path in [HPARAMS_LOG_DIR, SB3_LOG_DIR]:
        try:
            shutil.rmtree(path)
        except FileNotFoundError:
            pass
        except OSError as e:
            print(f"Warning: Could not remove directory {path}. {e}")

    db_file_path = os.path.join(OPTUNA_FOLDER, BASE_NAME + ".db")
    try:
        os.remove(db_file_path)
    except FileNotFoundError:
        pass
    except OSError as e:
        print(f"Warning: Could not remove database file {db_file_path}. {e}")

    print(f"Database file path will be: {DATABASE_FILE}")

    study = optuna.create_study(
        study_name=STUDY_NAME,
        direction="minimize",
        pruner=MedianPruner(),
        storage=DATABASE_FILE,
        load_if_exists=True,
    )

    with tf.summary.create_file_writer(HPARAMS_LOG_DIR).as_default():
        hp.hparams_config(
            hparams=[
                HP_CNN_KERNEL_SIZE, HP_CNN_FILTERS_LAYER1, HP_CNN_FILTERS_LAYER2,
                HP_TRANSFORMER_BLOCKS, HP_TRANSFORMER_HEADS, HP_TRANSFORMER_FF_DIM,
                HP_LR, HP_CLIP_RANGE, HP_GAE_LAMBDA, HP_N_EPOCHS,
                HP_ENT_COEF, HP_VF_COEF, HP_BATCH_SIZE, HP_N_STEPS
            ],
            metrics=[hp.Metric(METRIC_NAME, display_name='SoC Diff Abs Sum')]
        )

    try:
        study.optimize(
            objective,
            n_trials=20,
            callbacks=[],
            gc_after_trial=True,
        )
    except KeyboardInterrupt:
        print("Optimization interrupted by user.")
    except Exception as e:
        print(f"An unexpected error occurred during optimization: {e}")
        import traceback
        traceback.print_exc()

    print("Number of finished trials: ", len(study.trials))
    if study.best_trial:
        print("Best trial:")
        trial = study.best_trial
        print("  Value (Minimized SoC Diff Abs Sum): ", trial.value)
        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
    else:
        print("No trials finished successfully to determine best trial.")

    print("\n--- HParams Dashboard Visualization ---")
    print(f"To visualize the hyperparameter tuning results, run TensorBoard with the command:")
    print(f"%tensorboard --logdir {HPARAMS_LOG_DIR}")
    print("Then, navigate to the 'HParams' dashboard in TensorBoard.")
