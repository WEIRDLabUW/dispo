import os
from abc import ABC, abstractmethod

import gym
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from flax.core import FrozenDict
from flax.training import orbax_utils
from torch.utils.data import Dataset
from tqdm.rich import trange

from .datasets import OfflineDataset


class FeatureDataset(Dataset):
    def __init__(self, dataset, env):
        assert isinstance(dataset, OfflineDataset)
        assert isinstance(env, FeatureWrapper)
        self.dataset = dataset

        # Compute features for dataset
        features = []
        next_features = []
        batch_size = 1024
        for i in trange(0, len(self.dataset), batch_size, desc="Featurizing dataset"):
            if i + batch_size > len(self.dataset):
                batch_size = len(self.dataset) - i
            reward = self.dataset.rewards[i : i + batch_size]
            obs = self.dataset.observations[i : i + batch_size]
            next_obs = self.dataset.next_observations[i : i + batch_size]
            features.append(env.feature(obs, reward))
            next_features.append(env.feature(next_obs, reward))
        self.features = np.concatenate(features, axis=0).astype(np.float32)
        self.next_features = np.concatenate(next_features, axis=0).astype(np.float32)

        # Compute statistics for normalization
        self.feature_min = np.min(self.features, axis=0)
        self.feature_max = np.max(self.features, axis=0)

        # Normalize the bias separately
        self.feature_min[-1] = 0
        self.feature_max[-1] = 1

    def __getattr__(self, name):
        return getattr(self.dataset, name)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        data["features"] = self.features[idx]
        data["next_features"] = self.next_features[idx]
        return data

    def infer_reward_weights(self, num_reward_samples):
        # Sample random subset of data
        indices = np.random.permutation(len(self))[:num_reward_samples]
        
        # Perform least squares regression
        x = self.next_features[indices]
        y = self.rewards[indices]
        z = np.linalg.lstsq(x, y, rcond=None)[0]
        return z
        

class FeatureWrapper(gym.Wrapper, ABC):
    @property
    @abstractmethod
    def feat_dim(self):
        pass

    @abstractmethod
    def feature(
        self,
        obs: np.ndarray,
        reward: np.ndarray,
    ) -> np.ndarray:
        """
        Compute features for a batch of observations and rewards.
        """
        pass

    def step(self, action):
        obs, original_reward, done, info = self.env.step(action)
        reward = self.feature(obs[None], np.array([original_reward])[None])[0]
        info["original_reward"] = original_reward
        return obs, reward, done, info


class MLP(nn.Module):
    out_dim: int
    hidden_dim: int = 1024

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.swish(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.swish(x)
        x = nn.Dense(self.out_dim)(x)
        return x


class LearnedFeatureWrapper(FeatureWrapper, ABC):
    def __init__(self, env, raw_feat_dim, seed=0):
        super().__init__(env)
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]
        self.raw_feat_dim = raw_feat_dim

        self.build_models(seed)
        self.update_fn = self.get_update_fn()
        self.checkpointer = ocp.PyTreeCheckpointer()

        # Try loading saved feature net
        try:
            self.load()
        except:
            print("No saved feature network found")

    @property
    def feat_dim(self):
        return self.raw_feat_dim + 1

    @property
    @abstractmethod
    def save_path(self):
        pass

    @abstractmethod
    def build_models(self):
        pass

    @abstractmethod
    def get_update_fn(self):
        pass

    def train(self, obs, act, next_obs):
        self.params, self.opt_state, info = self.update_fn(
            self.params, self.opt_state, obs, act, next_obs
        )
        return info

    def save(self):
        ckpt = {"params": self.params}
        save_args = orbax_utils.save_args_from_target(ckpt)
        self.checkpointer.save(self.save_path, ckpt, save_args=save_args, force=True)

    def load(self):
        ckpt = self.checkpointer.restore(self.save_path)
        self.params = ckpt["params"]


class LaplacianFeatureWrapper(LearnedFeatureWrapper):
    @property
    def save_path(self):
        return os.path.join(
            "feature_nets", self.env.spec.id, f"laplacian_{self.raw_feat_dim}"
        )

    def build_models(self, seed):
        self.net = MLP(self.raw_feat_dim)
        self.params = self.net.init(
            jax.random.PRNGKey(seed), jnp.zeros((1, self.obs_dim))
        )["params"]
        self.tx = optax.adamw(learning_rate=1e-4)
        self.opt_state = self.tx.init(self.params)

    def feature(self, obs, reward):
        feat = self.net.apply({"params": self.params}, obs)
        return np.concatenate([feat, np.ones((len(obs), 1))], axis=-1)

    def get_update_fn(self):
        @jax.jit
        def update_fn(params, opt_state, obs, act, next_obs):
            def loss_fn(params, obs, next_obs):
                phi = self.net.apply({"params": params}, obs)
                next_phi = self.net.apply({"params": params}, next_obs)

                id_loss = ((phi - next_phi) ** 2).mean()

                cov = phi @ phi.T
                orho_loss_diag = -2 * cov.diagonal().mean()
                off_diag = ~jnp.eye(cov.shape[0], dtype=bool)
                orho_loss_off_diag = (
                    jnp.where(off_diag, cov, 0) ** 2
                ).sum() / off_diag.sum()

                loss = id_loss + orho_loss_diag + orho_loss_off_diag
                return loss

            # Compute gradients
            loss, grads = jax.value_and_grad(loss_fn)(params, obs, next_obs)

            # Apply gradient update
            updates, new_opt_state = self.tx.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            return new_params, new_opt_state, {"loss": loss}

        return update_fn


class DynamicsFeatureWrapper(LearnedFeatureWrapper):
    @property
    def save_path(self):
        return os.path.join(
            "feature_nets", self.env.spec.id, f"dynamics_{self.raw_feat_dim}"
        )

    def build_models(self, seed):
        rng = jax.random.PRNGKey(seed)
        self.phi = MLP(self.raw_feat_dim)
        rng, step_rng = jax.random.split(rng)
        phi_params = self.phi.init(step_rng, jnp.zeros((1, self.obs_dim)))["params"]

        self.dynamics = MLP(self.obs_dim)
        rng, step_rng = jax.random.split(rng)
        dynamics_params = self.dynamics.init(
            step_rng, jnp.zeros((1, self.raw_feat_dim + self.act_dim))
        )["params"]

        self.params = FrozenDict({"phi": phi_params, "dynamics": dynamics_params})
        self.tx = optax.adamw(learning_rate=1e-4)
        self.opt_state = self.tx.init(self.params)

    def feature(self, obs, reward):
        feat = self.phi.apply({"params": self.params["phi"]}, obs)
        return np.concatenate([feat, np.ones((len(obs), 1))], axis=-1)

    def get_update_fn(self):
        @jax.jit
        def update_fn(params, opt_state, obs, act, next_obs):
            def loss_fn(params, obs, act, next_obs):
                phi = self.phi.apply({"params": params["phi"]}, obs)
                next_obs_hat = self.dynamics.apply(
                    {"params": params["dynamics"]}, jnp.concatenate([phi, act], axis=-1)
                )
                loss = ((next_obs_hat - next_obs) ** 2).mean()
                return loss

            # Compute gradients
            loss, grads = jax.value_and_grad(loss_fn)(params, obs, act, next_obs)

            # Apply gradient update
            updates, new_opt_state = self.tx.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            return new_params, new_opt_state, {"loss": loss}

        return update_fn


class RandMLP(nn.Module):
    out_dim: int
    hidden_dim: int = 32
    normalize: bool = False

    @nn.compact
    def __call__(self, x):
        x = x[:, None, :]
        x = nn.Conv(self.hidden_dim * self.out_dim, (1,))(x)
        x = nn.relu(x)
        x = nn.Conv(
            self.hidden_dim * self.out_dim, (1,), feature_group_count=self.out_dim
        )(x)
        x = nn.relu(x)
        x = nn.Conv(self.out_dim, (1,), feature_group_count=self.out_dim)(x)
        if self.normalize:
            x = nn.tanh(x)
        return x[:, 0, :]


class RandomFeatureWrapper(FeatureWrapper):
    def __init__(self, env, rand_feat_dim, seed=0):
        super().__init__(env)
        self.in_dim = env.observation_space.shape[0]
        self.rand_feat_dim = rand_feat_dim

        self.net = RandMLP(self.rand_feat_dim)
        self.params = self.net.init(
            jax.random.PRNGKey(seed), jnp.zeros((1, self.in_dim))
        )
        self.params = jax.lax.stop_gradient(self.params)

    @property
    def feat_dim(self):
        return self.rand_feat_dim + 1

    def feature(self, obs, reward):
        feat = self.net.apply(self.params, obs)
        return np.concatenate([feat, np.ones((len(obs), 1))], axis=-1)


class FourierFeatureWrapper(RandomFeatureWrapper):
    def __init__(self, env, rand_feat_dim, seed=0):
        assert rand_feat_dim % 2 == 0, "Fourier features must have even dimension"
        super().__init__(env, rand_feat_dim // 2, seed)

    @property
    def feat_dim(self):
        return self.rand_feat_dim * 2 + 1

    def feature(self, obs, reward):
        feat = self.net.apply(self.params, obs)
        return np.concatenate(
            [np.sin(feat), np.cos(feat), np.ones((len(obs), 1))], axis=-1
        )


class PolynomialFeatureWrapper(FeatureWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.in_dim = env.observation_space.shape[0]

    @property
    def feat_dim(self):
        return int((self.in_dim + 2) * (self.in_dim + 1) / 2)

    def feature(self, obs, reward):
        x = np.ones((len(obs), self.in_dim + 1))
        x[:, :-1] = obs
        x = x[:, None, :] * x[:, :, None]
        triu_inds = np.triu_indices(self.in_dim + 1)
        return x[:, triu_inds[0], triu_inds[1]]


class DummyFeatureWrapper(FeatureWrapper):
    @property
    def feat_dim(self):
        return 1

    def feature(self, obs, reward):
        return reward
