import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import MeanFlowActor, Value


class MeanFlowFQLAgent(flax.struct.PyTreeNode):
    """FQL critic with a single native MeanFlow actor."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def critic_loss(self, batch, grad_params, rng):
        """Compute the TD critic loss with the same semantics as FQL."""
        rng, sample_rng = jax.random.split(rng)
        next_actions = self.sample_actions(batch['next_observations'], seed=sample_rng)
        next_actions = jnp.clip(next_actions, -1, 1)

        next_qs = self.network.select('target_critic')(batch['next_observations'], actions=next_actions)
        if self.config['q_agg'] == 'min':
            next_q = next_qs.min(axis=0)
        else:
            next_q = next_qs.mean(axis=0)

        target_q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_q

        q = self.network.select('critic')(batch['observations'], actions=batch['actions'], params=grad_params)
        critic_loss = jnp.square(q - target_q).mean()

        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
        }

    def _meanflow_action(self, observations, noises, params=None):
        t = jnp.ones((*noises.shape[:-1], 1), dtype=jnp.float32)
        return self.network.select('actor')(observations, noises, t, params=params)

    def actor_loss(self, batch, grad_params, rng):
        """Compute MeanFlow JVP and Q losses."""
        batch_size, action_dim = batch['actions'].shape
        rng, noise_rng, time_rng, q_rng, mse_rng = jax.random.split(rng, 5)

        # MeanFlow convention: t=0 is the dataset action and t=1 is Gaussian noise.
        actions = batch['actions']
        noises = jax.random.normal(noise_rng, (batch_size, action_dim))
        num_time_steps = int(self.config['meanflow_time_steps'])
        time_idxs = jax.random.randint(time_rng, (batch_size, 1), 0, num_time_steps)
        t = (time_idxs.astype(jnp.float32) + 1.0) / jnp.asarray(num_time_steps, dtype=jnp.float32)
        z_t = (1.0 - t) * actions + t * noises
        vel = noises - actions

        def actor_fn(z_input, t_input):
            return self.network.select('actor')(
                batch['observations'],
                z_input,
                t_input,
                params=grad_params,
            )

        g, dgdt = jax.jvp(actor_fn, (z_t, t), (vel, jnp.ones_like(t)))
        g_target = z_t + (t - 1.0) * vel - t * dgdt
        g_target = jax.lax.stop_gradient(g_target)
        meanflow_loss = jnp.mean((g - g_target) ** 2)

        q_noises = jax.random.normal(q_rng, (batch_size, action_dim))
        actor_actions = self._meanflow_action(batch['observations'], q_noises, params=grad_params)
        actor_actions = jnp.clip(actor_actions, -1, 1)
        qs = self.network.select('critic')(batch['observations'], actions=actor_actions)
        q = jnp.mean(qs, axis=0)

        q_loss = -q.mean()
        if self.config['normalize_q_loss']:
            lam = jax.lax.stop_gradient(1.0 / (jnp.abs(q).mean() + 1e-8))
            q_loss = lam * q_loss

        actor_loss = self.config['alpha'] * meanflow_loss + q_loss

        mse_actions = self.sample_actions(batch['observations'], seed=mse_rng)
        mse = jnp.mean((mse_actions - batch['actions']) ** 2)

        return actor_loss, {
            'actor_loss': actor_loss,
            'meanflow_loss': meanflow_loss,
            'q_loss': q_loss,
            'q': q.mean(),
            'mse': mse,
            'jvp_norm': jnp.linalg.norm(dgdt, axis=-1).mean(),
            'target_norm': jnp.linalg.norm(g_target, axis=-1).mean(),
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        rng, actor_rng, critic_rng = jax.random.split(rng, 3)

        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = critic_loss + actor_loss
        return loss, info

    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @jax.jit
    def update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'critic')

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(
        self,
        observations,
        seed=None,
        temperature=1.0,
    ):
        """Sample actions with one actor call: action = actor(s, noise, 1)."""
        del temperature
        seed, action_seed = jax.random.split(seed)
        noises = jax.random.normal(
            action_seed,
            (
                *observations.shape[: -len(self.config['ob_dims'])],
                self.config['action_dim'],
            ),
        )
        actions = self._meanflow_action(observations, noises)
        actions = jnp.clip(actions, -1, 1)
        return actions

    @classmethod
    def create(
        cls,
        seed,
        example_batch,
        config,
    ):
        """Create a new agent."""
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_observations = example_batch['observations']
        ex_actions = example_batch['actions']
        ex_times = ex_actions[..., :1]
        ob_dims = ex_observations.shape[1:]
        action_dim = ex_actions.shape[-1]

        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['critic'] = encoder_module()
            encoders['actor'] = encoder_module()

        critic_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=2,
            encoder=encoders.get('critic'),
        )
        actor_def = MeanFlowActor(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor'),
        )

        network_info = dict(
            critic=(critic_def, (ex_observations, ex_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, ex_actions)),
            actor=(actor_def, (ex_observations, ex_actions, ex_times)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params['modules_target_critic'] = params['modules_critic']

        config['ob_dims'] = ob_dims
        config['action_dim'] = action_dim
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    return ml_collections.ConfigDict(
        dict(
            agent_name='meanflow_fql',
            ob_dims=ml_collections.config_dict.placeholder(list),
            action_dim=ml_collections.config_dict.placeholder(int),
            lr=3e-4,
            batch_size=256,
            actor_hidden_dims=(512, 512, 512, 512),
            value_hidden_dims=(512, 512, 512, 512),
            layer_norm=True,
            actor_layer_norm=False,
            discount=0.99,
            tau=0.005,
            q_agg='mean',
            alpha=10.0,
            normalize_q_loss=False,
            encoder=ml_collections.config_dict.placeholder(str),
            meanflow_time_steps=50,
        )
    )
