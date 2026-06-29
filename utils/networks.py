import functools
from typing import Any, Optional, Sequence, Type

import distrax
import flax.linen as nn
import jax.numpy as jnp


def default_init(scale=1.0):
    """Default kernel initializer."""
    return nn.initializers.variance_scaling(scale, 'fan_avg', 'uniform')


def ensemblize(cls, num_qs, in_axes=None, out_axes=0, **kwargs):
    """Ensemblize a module."""
    return nn.vmap(
        cls,
        variable_axes={'params': 0, 'intermediates': 0},
        split_rngs={'params': True},
        in_axes=in_axes,
        out_axes=out_axes,
        axis_size=num_qs,
        **kwargs,
    )


class MLP(nn.Module):
    """Multi-layer perceptron.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        activations: Activation function.
        activate_final: Whether to apply activation to the final layer.
        kernel_init: Kernel initializer.
        layer_norm: Whether to apply layer normalization.
    """

    hidden_dims: Sequence[int]
    activations: Any = nn.gelu
    activate_final: bool = False
    kernel_init: Any = default_init()
    layer_norm: bool = False
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, x, training=False):
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=self.kernel_init)(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = self.activations(x)
                if self.layer_norm:
                    x = nn.LayerNorm()(x)
                if self.dropout_rate > 0.0:
                    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not training)
            if i == len(self.hidden_dims) - 2:
                self.sow('intermediates', 'feature', x)
        return x


class LogParam(nn.Module):
    """Scalar parameter module with log scale."""

    init_value: float = 1.0

    @nn.compact
    def __call__(self):
        log_value = self.param('log_value', init_fn=lambda key: jnp.full((), jnp.log(self.init_value)))
        return jnp.exp(log_value)


class TransformedWithMode(distrax.Transformed):
    """Transformed distribution with mode calculation."""

    def mode(self):
        return self.bijector.forward(self.distribution.mode())


class Normal(nn.Module):
    """Gaussian policy head, optionally squashed by tanh."""

    base_cls: Type[nn.Module]
    action_dim: int
    log_std_min: Optional[float] = -20
    log_std_max: Optional[float] = 2
    learnable_log_std_multiplier: Optional[float] = None
    learnable_log_std_offset: Optional[float] = None
    state_dependent_std: bool = True
    squash_tanh: bool = False
    fixed_log_std: Optional[float] = None

    @nn.compact
    def __call__(self, inputs, *args, **kwargs):
        x = self.base_cls()(inputs, *args, **kwargs)

        means = nn.Dense(self.action_dim, kernel_init=default_init(), name='OutputDenseMean')(x)
        if self.state_dependent_std:
            log_stds = nn.Dense(self.action_dim, kernel_init=default_init(), name='OutputDenseLogStd')(x)
        else:
            log_stds = self.param('OutputLogStd', nn.initializers.zeros, (self.action_dim,), jnp.float32)

        if self.learnable_log_std_multiplier is not None:
            log_stds *= self.param(
                'LogStdMul',
                nn.initializers.constant(self.learnable_log_std_multiplier),
                (),
                jnp.float32,
            )
        if self.learnable_log_std_offset is not None:
            log_stds += self.param(
                'LogStdOffset',
                nn.initializers.constant(self.learnable_log_std_offset),
                (),
                jnp.float32,
            )

        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)
        if self.fixed_log_std is not None:
            log_stds = jnp.ones_like(log_stds) * self.fixed_log_std

        distribution = distrax.MultivariateNormalDiag(loc=means, scale_diag=jnp.exp(log_stds))
        if self.squash_tanh:
            return TransformedWithMode(distribution, distrax.Block(distrax.Tanh(), ndims=1))
        return distribution


TanhNormal = functools.partial(Normal, squash_tanh=True)


class CosineEmbedding(nn.Module):
    """Cosine embedding"""
    num_cosines: int = 64

    @nn.compact
    def __call__(self, taus):
        freqs = jnp.pi * jnp.arange(1, self.num_cosines + 1)[None, None]
        cos_embeddings = jnp.cos(freqs * taus[..., None])

        return cos_embeddings


def compute_support(q_min, q_max, num_bins):
    """Build evenly spaced support points for return probability embeddings."""
    return jnp.linspace(q_min, q_max, num_bins)


def to_probs(values, support, sigma):
    """Softly embed scalar returns over support bins."""
    logits = -((values[..., None] - support[:-1]) ** 2) / (2 * sigma**2)
    return nn.softmax(logits, axis=-1)


class ResMLP(nn.Module):
    """Small residual MLP used by the critic vector field."""

    per_transformation_hidden_dims: Sequence[int]
    num_transformations: int
    layer_norm: bool = False
    output_dim: int = 1
    activate_before_layernorm: bool = True

    @nn.compact
    def __call__(self, x):
        hidden_dim = self.per_transformation_hidden_dims[-1]
        x = nn.Dense(hidden_dim, kernel_init=default_init())(x)

        for _ in range(self.num_transformations):
            residual = x
            y = x
            for hidden_dim in self.per_transformation_hidden_dims:
                y = nn.Dense(hidden_dim, kernel_init=default_init())(y)
                if self.activate_before_layernorm:
                    y = nn.gelu(y)
                    if self.layer_norm:
                        y = nn.LayerNorm()(y)
                else:
                    if self.layer_norm:
                        y = nn.LayerNorm()(y)
                    y = nn.gelu(y)
            if residual.shape[-1] != y.shape[-1]:
                residual = nn.Dense(y.shape[-1], kernel_init=default_init())(residual)
            x = residual + y

        return nn.Dense(self.output_dim, kernel_init=default_init())(x)


class CriticResVectorField(nn.Module):
    """Critic vector field with return probability embedding and residual MLP."""

    per_transformation_hidden_dims: Sequence[int]
    num_transformations: int
    activate_before_layernorm: bool = True
    layer_norm: bool = False
    encoder: nn.Module = None
    output_dim: int = 1
    embed_time: bool = True
    use_prob_embed: bool = True
    q_min: float = 0.0
    q_max: float = 0.0
    sigma: float = 16.0
    num_bins: int = 51

    def setup(self) -> None:
        self.mlp = ResMLP(
            per_transformation_hidden_dims=self.per_transformation_hidden_dims,
            num_transformations=self.num_transformations,
            layer_norm=self.layer_norm,
            output_dim=self.output_dim,
            activate_before_layernorm=self.activate_before_layernorm,
        )

    @nn.compact
    def __call__(self, returns, start_times, end_times, observations, actions=None):
        if self.encoder is not None:
            observations = self.encoder(observations)

        if actions is None:
            state_action = observations
        else:
            state_action = jnp.concatenate([observations, actions], axis=-1)

        if self.use_prob_embed:
            support = compute_support(self.q_min, self.q_max, self.num_bins)
            bin_width = jnp.maximum(support[1] - support[0], 1e-6)
            returns_embed = to_probs(returns.squeeze(-1), support, self.sigma * bin_width)
        else:
            returns_embed = (returns - self.q_min) / (self.q_max - self.q_min + 1e-6)

        if self.embed_time:
            time_embed = jnp.concatenate(
                [jnp.cos(start_times), jnp.cos(end_times), end_times - start_times],
                axis=-1,
            )
        else:
            time_embed = jnp.concatenate(
                [start_times, end_times, end_times - start_times],
                axis=-1,
            )

        inputs = jnp.concatenate([state_action, returns_embed, time_embed], axis=-1)
        return self.mlp(inputs)


class Actor(nn.Module):
    """Gaussian actor network.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        action_dim: Action dimension.
        layer_norm: Whether to apply layer normalization.
        log_std_min: Minimum value of log standard deviation.
        log_std_max: Maximum value of log standard deviation.
        tanh_squash: Whether to squash the action with tanh.
        state_dependent_std: Whether to use state-dependent standard deviation.
        const_std: Whether to use constant standard deviation.
        final_fc_init_scale: Initial scale of the final fully-connected layer.
        encoder: Optional encoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    action_dim: int
    layer_norm: bool = False
    log_std_min: Optional[float] = -5
    log_std_max: Optional[float] = 2
    tanh_squash: bool = False
    state_dependent_std: bool = False
    const_std: bool = True
    final_fc_init_scale: float = 1e-2
    encoder: nn.Module = None

    def setup(self):
        self.actor_net = MLP(self.hidden_dims, activate_final=True, layer_norm=self.layer_norm)
        self.mean_net = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))
        if self.state_dependent_std:
            self.log_std_net = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))
        else:
            if not self.const_std:
                self.log_stds = self.param('log_stds', nn.initializers.zeros, (self.action_dim,))

    def __call__(
        self,
        observations,
        temperature=1.0,
    ):
        """Return action distributions.

        Args:
            observations: Observations.
            temperature: Scaling factor for the standard deviation.
        """
        if self.encoder is not None:
            inputs = self.encoder(observations)
        else:
            inputs = observations
        outputs = self.actor_net(inputs)

        means = self.mean_net(outputs)
        if self.state_dependent_std:
            log_stds = self.log_std_net(outputs)
        else:
            if self.const_std:
                log_stds = jnp.zeros_like(means)
            else:
                log_stds = self.log_stds

        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        distribution = distrax.MultivariateNormalDiag(loc=means, scale_diag=jnp.exp(log_stds) * temperature)
        if self.tanh_squash:
            distribution = TransformedWithMode(distribution, distrax.Block(distrax.Tanh(), ndims=1))

        return distribution


class Value(nn.Module):
    """Value/critic network.

    This module can be used for both value V(s, g) and critic Q(s, a, g) functions.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        value_dim: Value dimension.
        layer_norm: Whether to apply layer normalization.
        num_ensembles: Number of ensemble components.
        encoder: Optional encoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    value_dim: int = 1
    layer_norm: bool = True
    num_ensembles: int = 2
    encoder: nn.Module = None

    def setup(self):
        mlp_class = MLP
        if self.num_ensembles > 1:
            mlp_class = ensemblize(mlp_class, self.num_ensembles)
        value_net = mlp_class((*self.hidden_dims, self.value_dim),
                              activate_final=False, layer_norm=self.layer_norm)

        self.value_net = value_net

    def __call__(self, observations, actions=None):
        """Return values or critic values.

        Args:
            observations: Observations.
            actions: Actions (optional).
        """
        if self.encoder is not None:
            inputs = [self.encoder(observations)]
        else:
            inputs = [observations]
        if actions is not None:
            inputs.append(actions)
        inputs = jnp.concatenate(inputs, axis=-1)

        v = self.value_net(inputs)
        if self.value_dim == 1:
            v = v.squeeze(-1)

        return v


class ValueVectorField(nn.Module):
    """Value vector field network for flow matching.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        value_dim: Value dimension.
        layer_norm: Whether to apply layer normalization.
        encoder: Optional encoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    value_dim: int = 1
    layer_norm: bool = False
    num_ensembles: int = 2
    encoder: nn.Module = None

    def setup(self) -> None:
        mlp_class = MLP
        if self.num_ensembles > 1:
            mlp_class = ensemblize(mlp_class, self.num_ensembles)
        value_net = mlp_class((*self.hidden_dims, self.value_dim),
                              activate_final=False, layer_norm=self.layer_norm)

        self.value_net = value_net

    @nn.compact
    def __call__(self, returns, times, observations, actions=None):
        """Return the vectors at the given states, actions, and times.

        Args:
            returns: Returns.
            times: Times.
            observations: Observations.
            actions: Actions.
        """
        if self.encoder is not None:
            observations = self.encoder(observations)
        if actions is None:
            inputs = jnp.concatenate([returns, times, observations], axis=-1)
        else:
            inputs = jnp.concatenate([returns, times, observations, actions], axis=-1)

        v = self.value_net(inputs)

        return v


class FourierValueVectorField(nn.Module):
    """Value vector field with Fourier time embeddings."""

    hidden_dims: Sequence[int]
    value_dim: int = 1
    time_embedding_dim: int = 64
    layer_norm: bool = False
    num_ensembles: int = 2
    encoder: nn.Module = None

    def setup(self) -> None:
        mlp_class = MLP
        if self.num_ensembles > 1:
            mlp_class = ensemblize(mlp_class, self.num_ensembles)
        value_net = mlp_class((*self.hidden_dims, self.value_dim),
                              activate_final=False, layer_norm=self.layer_norm)

        self.time_embedding_net = MLP(
            (self.time_embedding_dim,), activate_final=False, layer_norm=self.layer_norm
        )
        self.value_net = value_net

    @nn.compact
    def __call__(self, returns, times, observations, actions=None):
        """Return value vector fields with Fourier-embedded times."""
        if self.encoder is not None:
            observations = self.encoder(observations)

        num_freqs = self.time_embedding_dim // 2
        freqs = jnp.pi * jnp.arange(1, num_freqs + 1, dtype=times.dtype)
        angles = times * freqs
        time_embedding = jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=-1)
        if self.time_embedding_dim % 2 == 1:
            time_embedding = jnp.concatenate([time_embedding, times], axis=-1)
        time_embedding = self.time_embedding_net(time_embedding)

        if actions is None:
            inputs = jnp.concatenate([returns, time_embedding, observations], axis=-1)
        else:
            inputs = jnp.concatenate([returns, time_embedding, observations, actions], axis=-1)

        v = self.value_net(inputs)

        return v


class ActorVectorField(nn.Module):
    """Actor vector field network for flow matching.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        action_dim: Action dimension.
        layer_norm: Whether to apply layer normalization.
        encoder: Optional encoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    action_dim: int
    layer_norm: bool = False
    encoder: nn.Module = None
    need_time: bool = True
    encode_time_dim: Optional[int] = None

    def setup(self) -> None:
        self.mlp = MLP((*self.hidden_dims, self.action_dim),
                       activate_final=False, layer_norm=self.layer_norm)

    @nn.compact
    def __call__(self, observations, actions, times=None, end_times=None, is_encoded=False):
        """Return the vectors at the given states, actions, and times (optional).

        Args:
            observations: Observations.
            actions: Actions.
            times: Start times (optional).
            end_times: End times (optional, for MeanFlow average velocities).
            is_encoded: Whether the observations are already encoded.
        """
        if not is_encoded and self.encoder is not None:
            observations = self.encoder(observations)
        if times is None:
            inputs = jnp.concatenate([observations, actions], axis=-1)
        elif end_times is None:
            inputs = jnp.concatenate([observations, actions, times], axis=-1)
        else:
            time_embed = jnp.concatenate([times, end_times, end_times - times], axis=-1)
            inputs = jnp.concatenate([observations, actions, time_embed], axis=-1)

        v = self.mlp(inputs)

        return v


class ShortcutActorVectorField(nn.Module):
    """Shortcut actor that predicts an average velocity over a step size."""

    hidden_dims: Sequence[int]
    action_dim: int
    layer_norm: bool = False
    encoder: nn.Module = None

    def setup(self) -> None:
        self.mlp = MLP((*self.hidden_dims, self.action_dim), activate_final=False, layer_norm=self.layer_norm)

    @nn.compact
    def __call__(self, observations, noisy_actions, times, step_sizes):
        if self.encoder is not None:
            observations = self.encoder(observations)
        inputs = jnp.concatenate([observations, noisy_actions, times, step_sizes], axis=-1)
        return self.mlp(inputs)


class MeanFlowActor(nn.Module):
    """MeanFlow actor that predicts a clean action from z_t and t."""

    hidden_dims: Sequence[int]
    action_dim: int
    layer_norm: bool = False
    encoder: nn.Module = None

    def setup(self) -> None:
        self.mlp = MLP((*self.hidden_dims, self.action_dim), activate_final=False, layer_norm=self.layer_norm)

    @nn.compact
    def __call__(self, observations, noisy_actions, times):
        if self.encoder is not None:
            observations = self.encoder(observations)
        inputs = jnp.concatenate([observations, noisy_actions, times], axis=-1)
        return self.mlp(inputs)


class ImplicitQuantileValue(nn.Module):
    """implicit quantile value/critic network.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        tau_embedding_num_cosines: Number of cosines in the tau embedding.
        embedding_dim: Embedding dimension.
        layer_norm: Whether to apply layer normalization.
        encoder: Optional encoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    tau_embedding_num_cosines: int
    embedding_dim: int
    layer_norm: bool = False
    num_ensembles: int = 2
    encoder: nn.Module = None

    def setup(self) -> None:
        mlp_class = MLP
        self.tau_cosine_embedding = CosineEmbedding(
            num_cosines=self.tau_embedding_num_cosines)
        self.tau_embedding_net = mlp_class(
            (self.embedding_dim,), activate_final=False, layer_norm=self.layer_norm)
        self.sa_embedding_net = mlp_class(
            (self.embedding_dim,), activate_final=False, layer_norm=self.layer_norm)

        if self.num_ensembles > 1:
            mlp_class = ensemblize(mlp_class, self.num_ensembles)
        value_net = mlp_class((*self.hidden_dims, 1), activate_final=False, layer_norm=self.layer_norm)
        self.value_net = value_net

    @nn.compact
    def __call__(self, observations, actions, taus):
        if self.encoder is not None:
            observations = self.encoder(observations)
        tau_cosine_embeddings = self.tau_cosine_embedding(taus)
        tau_embeddings = self.tau_embedding_net(tau_cosine_embeddings)
        sa_embeddings = self.sa_embedding_net(
            jnp.concatenate([observations, actions], axis=-1))
        embeddings = (jnp.expand_dims(sa_embeddings, axis=-2) * tau_embeddings)  # (num_ensembles, batch_size, ..., num_taus, embedding_dims)
        quantiles = self.value_net(embeddings)  # (num_ensembles, batch_size, ..., num_taus, 1)
        
        return quantiles
