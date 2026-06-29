# Copyright (c) 2025-2026, Inria
# Adapted from the original fql.py made by the FQL authors.

import copy
from typing import Any
from functools import partial

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax
from jax.scipy.special import logsumexp

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ActorVectorField, Value
from utils.datasets import GPU_dataset_sample


class GFPAgent(flax.struct.PyTreeNode):
    """Guided Flow Policy (GFP) agent."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def critic_eval(self, obs, actions):
        """Compute Q(s,a) with the target network and q aggregation function."""
        q = self.network.select("target_critic")(obs, actions)
        return getattr(jnp, self.config["q_agg"])(q, axis=0)

    def value_with_all_options(self, agg_rule, obs, actor_acts, vabc_acts):
        """
        Evaluate the value of a state V(s), with a switch-case depending
        on whether we use the Actor to compute the reference action
        or the VaBC policy, or both (with an aggregate function).

        Takes as input lambda functions for lazy computations.
        """
        assert agg_rule in ["actor", "vabc", "min", "mean", "max"]
        if agg_rule != "vabc":
            q_actor = self.critic_eval(obs, actor_acts)
        if agg_rule != "actor":
            q_vabc = self.critic_eval(obs, vabc_acts)
        if agg_rule == "actor":
            return q_actor
        elif agg_rule == "vabc":
            return q_vabc
        elif agg_rule == "max":
            return jnp.maximum(q_actor, q_vabc)
        elif agg_rule == "min":
            return jnp.minimum(q_actor, q_vabc)
        else:
            return (q_actor + q_vabc) / 2

    def critic_loss(self, batch, grad_params, rng):
        """Compute GPS's critic loss"""
        rng, sample_rng = jax.random.split(rng)
        next_obs = batch["next_observations"]
        if self.config["target_use_next"]:
            vabc_obs = next_obs
        else:
            vabc_obs = batch["observations"]
        tgt_agg = self.config["target_agg"]

        # Compute the Bellman target with different rules
        # Lazy computations
        if tgt_agg != "actor":
            vabc_acts = self.sample_flow_actions(vabc_obs, sample_rng)
        else:
            vabc_acts = None
        if tgt_agg != "vabc":
            actor_acts = self.sample_actions(next_obs, sample_rng)
        else:
            actor_acts = None
        next_q = self.value_with_all_options(tgt_agg, next_obs, actor_acts, vabc_acts)
        target_q = batch["rewards"] + self.config["discount"] * batch["masks"] * next_q

        q = self.network.select("critic")(
            batch["observations"], actions=batch["actions"], params=grad_params
        )
        critic_loss = jnp.square(q - target_q).mean()

        # Metrics:
        if self.config["log_metrics"]:
            info = {
                "critic_loss": critic_loss,
                "q_mean": q.mean(),
                "tgt_q_mean": next_q.mean(),
                "q_max": q.max(),
                "q_min": q.min(),
            }
        else:
            info = dict()
        return critic_loss, info

    def compute_guidance(self, batch, actor_acts, vabc_acts, lam):
        """Compute GFP-VaBC guidance weights"""
        obs = batch["observations"]

        # Compute the reference value with different rules
        q_ref = self.value_with_all_options(
            self.config["guidance_agg"], obs, actor_acts, vabc_acts
        )
        q_data = self.critic_eval(obs, batch["actions"])
        scale = lam / self.config["eta_temperature"]

        # Compute the guidance, either using a softmax or as an advantage
        assert self.config["guidance_fn"] in ["softmax", "advantage"]
        scl_q_data = scale * q_data
        scl_q_ref = scale * q_ref
        if self.config["guidance_fn"] == "softmax":
            log_denominator = logsumexp(
                jnp.stack([scl_q_data, scl_q_ref], axis=-1), axis=-1
            )
            guidance = jnp.exp(scl_q_data - log_denominator)
        else:
            guidance = jnp.exp(scl_q_data - scl_q_ref)

        # Metrics:
        if self.config["log_metrics"]:
            info = {
                "q_data": q_data.mean(),
                "q_ref": q_ref.mean(),
                "scaled_q_data": scl_q_data.mean(),
                "scaled_q_ref": scl_q_ref.mean(),
                "guidance": guidance.mean(),
                "guidance_above_01": jnp.mean(guidance > 0.01),
                "guidance_above_10": jnp.mean(guidance > 0.1),
                "guidance_above_25": jnp.mean(guidance > 0.25),
                "guidance_above_40": jnp.mean(guidance > 0.4),
                "guidance_above_50": jnp.mean(guidance > 0.5),
                "guidance_above_75": jnp.mean(guidance > 0.75),
            }
        else:
            info = dict()
        return guidance, info

    def actor_loss(self, batch, grad_params, rng):
        """Compute both the actor and the VaBC losses"""
        batch_size, action_dim = batch["actions"].shape
        rng, distill_rng, flow_x_rng, flow_t_rng = jax.random.split(rng, 4)

        # Distillation loss
        noises = jax.random.normal(distill_rng, (batch_size, action_dim))
        target_vabc_acts = self.compute_flow_actions(
            batch["observations"], noises=noises
        )
        actor_acts = self.network.select("actor")(
            batch["observations"], noises, params=grad_params
        )
        distill_loss = jnp.mean((actor_acts - target_vabc_acts) ** 2)

        # Q loss
        actor_acts = jnp.clip(actor_acts, -1, 1)
        qs = self.network.select("critic")(batch["observations"], actions=actor_acts)
        q = getattr(jnp, self.config["q_agg"])(qs, axis=0)
        q_loss = -q.mean()
        lam = jax.lax.stop_gradient(1 / jnp.abs(q).mean())
        if self.config["normalize_q_loss"]:
            q_loss = lam * q_loss

        # == VaBC loss ==
        # Noising
        x_0 = jax.random.normal(flow_x_rng, (batch_size, action_dim))
        x_1 = batch["actions"]
        t = jax.random.uniform(flow_t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        # Guidance * error
        pred = self.network.select("vabc")(
            batch["observations"], x_t, t, params=grad_params
        )
        flow_error = jnp.mean((pred - vel) ** 2, axis=1)
        if self.config["eta_temperature"] == 0:
            VaBC_loss = jnp.mean(flow_error)
            g_info = dict()
        else:
            guidance, g_info = self.compute_guidance(
                batch, jax.lax.stop_gradient(actor_acts), target_vabc_acts, lam
            )
            VaBC_loss = jnp.mean(guidance * flow_error)

        # Total loss
        actor_loss = VaBC_loss + self.config["alpha"] * distill_loss + q_loss
        if self.config["log_metrics"]:
            info = dict(
                actor_loss=actor_loss,
                VaBC_loss=VaBC_loss,
                distill_loss=distill_loss,
                q_loss=q_loss,
                **g_info,
            )
        else:
            info = dict()
        return actor_loss, info

    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        rng, actor_rng, critic_rng = jax.random.split(rng, 3)

        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f"critic/{k}"] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f"actor/{k}"] = v

        loss = critic_loss + actor_loss
        return loss, info

    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config["tau"] + tp * (1 - self.config["tau"]),
            self.network.params[f"modules_{module_name}"],
            self.network.params[f"modules_target_{module_name}"],
        )
        network.params[f"modules_target_{module_name}"] = new_target_params

    def switch_config_to_online(self):
        new_config = self.config.copy(
            {"eta_temperature": self.config["eta_temperature_online"]}
        )
        return self.replace(config=new_config)

    @jax.jit
    def update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        rng, loss_rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=loss_rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, "critic")

        return self.replace(network=new_network, rng=rng), info

    @partial(jax.jit, static_argnames=("n_steps_fused", "dataset_size", "batch_size"))
    def multi_sample_and_update(
        self,
        n_steps_fused,
        dataset,
        dataset_size,
        batch_size,
    ):
        _agent = self
        for _ in range(n_steps_fused):
            batch, rng = GPU_dataset_sample(
                _agent.rng, dataset, batch_size, dataset_size
            )
            rng, loss_rng = jax.random.split(rng)

            def loss_fn(grad_params):
                return _agent.total_loss(batch, grad_params, rng=loss_rng)

            new_network, info = _agent.network.apply_loss_fn(loss_fn=loss_fn)
            _agent.target_update(new_network, "critic")

            _agent = _agent.replace(network=new_network, rng=rng)
        return _agent, info

    def sample_noises(self, obs, seed):
        return jax.random.normal(
            seed,
            (
                *obs.shape[: -len(self.config["ob_dims"])],  # (B,) or (,)
                self.config["action_dim"],
            ),
        )

    @jax.jit
    def sample_actions(self, observations, seed, temperature=1.0):
        """Sample actions from the one-step actor policy."""
        noises = self.sample_noises(observations, seed)
        actions = self.network.select("actor")(observations, noises)
        actions = jnp.clip(actions, -1, 1)
        return actions

    def compute_flow_actions(
        self,
        observations,
        noises,
    ):
        """Compute actions from the VaBC flow model using the Euler method."""
        if self.config["encoder"] is not None:
            observations = self.network.select("vabc_encoder")(observations)
        actions = noises
        # Euler method
        for i in range(self.config["flow_steps"]):
            t = jnp.full((*observations.shape[:-1], 1), i / self.config["flow_steps"])
            vels = self.network.select("vabc")(
                observations, actions, t, is_encoded=True
            )
            actions = actions + vels / self.config["flow_steps"]
        actions = jnp.clip(actions, -1, 1)
        return actions

    @jax.jit
    def sample_flow_actions(self, observations, seed, temperature=1.0):
        """Sample initial noise and call compute_flow_actions to get VaBC actions"""
        noises = self.sample_noises(observations, seed)
        return self.compute_flow_actions(observations, noises)

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
    ):
        """Create a new agent.

        Args:
            seed: Random seed.
            ex_observations: Example batch of observations.
            ex_actions: Example batch of actions.
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_times = ex_actions[..., :1]
        ob_dims = ex_observations.shape[1:]
        action_dim = ex_actions.shape[-1]
        # Define encoders.
        encoders = dict()
        if config["encoder"] is not None:
            encoder_module = encoder_modules[config["encoder"]]
            encoders["critic"] = encoder_module()
            encoders["vabc"] = encoder_module()
            encoders["actor"] = encoder_module()

        # Define networks.
        critic_def = Value(
            hidden_dims=config["value_hidden_dims"],
            layer_norm=config["layer_norm"],
            num_ensembles=2,
            encoder=encoders.get("critic"),
        )
        vabc_def = ActorVectorField(
            hidden_dims=config["actor_hidden_dims"],
            action_dim=action_dim,
            layer_norm=config["actor_layer_norm"],
            encoder=encoders.get("vabc"),
            need_time=True,
            encode_time_dim=config["encode_time_dim"],
        )
        actor_def = ActorVectorField(
            hidden_dims=config["actor_hidden_dims"],
            action_dim=action_dim,
            layer_norm=config["actor_layer_norm"],
            encoder=encoders.get("actor"),
            need_time=False,
        )

        network_info = dict(
            critic=(critic_def, (ex_observations, ex_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, ex_actions)),
            vabc=(vabc_def, (ex_observations, ex_actions, ex_times)),
            actor=(actor_def, (ex_observations, ex_actions)),
        )

        if encoders.get("vabc") is not None:
            # Add vabc_encoder to ModuleDict to make it separately callable.
            network_info["vabc_encoder"] = (encoders.get("vabc"), (ex_observations,))
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config["lr"])
        network_params = network_def.init(init_rng, **network_args)["params"]
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params["modules_target_critic"] = params["modules_critic"]

        config["ob_dims"] = ob_dims
        config["action_dim"] = action_dim
        if config["eta_temperature_online"] is None:
            config["eta_temperature_online"] = config["eta_temperature"]
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='gfp',
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
            flow_steps=10,
            normalize_q_loss=False,
            target_use_next=True,
            target_agg='actor',
            guidance_agg='actor',
            guidance_fn='softmax',
            eta_temperature=1.0,
            eta_temperature_online=ml_collections.config_dict.placeholder(float),
            log_metrics=True,
            encode_time_dim=0,
            encoder=ml_collections.config_dict.placeholder(str),
        )
    )
    return config
