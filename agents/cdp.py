import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ActorVectorField, Value


# ============================================================
# Core: Compute Drift V and Loss
# ============================================================

def compute_drift(
    gen: jnp.ndarray,
    pos: jnp.ndarray,
    neg: jnp.ndarray,
    temp: float = 0.05,
    exclude_self_neg: bool = False,
    pos_logit_bias=None,
    neg_logit_bias=None,
):
    """Compute drift field V with explicit positive and negative sets.

    Args:
        gen: Generated samples [..., G, D]
        pos: Positive samples [..., P, D]
        neg: Negative samples [..., N, D]
        temp: Temperature for softmax kernel.
        exclude_self_neg: Whether to mask diagonal in gen-neg distances.
        pos_logit_bias: Optional bias added to positive logits, broadcastable to [..., G, P].
        neg_logit_bias: Optional bias added to negative logits, broadcastable to [..., G, N].

    Returns:
        V: Drift vectors [..., G, D]
    """
    dist_pos = jnp.linalg.norm(gen[..., :, None, :] - pos[..., None, :, :], axis=-1)  # [..., G, P]
    dist_neg = jnp.linalg.norm(gen[..., :, None, :] - neg[..., None, :, :], axis=-1)  # [..., G, N]

    if exclude_self_neg:
        g = gen.shape[-2]
        n = neg.shape[-2]
        dist_neg = dist_neg + jnp.eye(g, n, dtype=dist_neg.dtype) * 1e6

    logits_pos = -dist_pos / temp
    logits_neg = -dist_neg / temp
    if pos_logit_bias is not None:
        logits_pos = logits_pos + pos_logit_bias
    if neg_logit_bias is not None:
        logits_neg = logits_neg + neg_logit_bias
    logits = jnp.concatenate([logits_pos, logits_neg], axis=-1)  # [..., G, P+N]

    a_row = jax.nn.softmax(logits, axis=-1)
    a_col = jax.nn.softmax(logits, axis=-2)
    a = jnp.sqrt(jnp.clip(a_row * a_col, a_min=1e-12))

    p = pos.shape[-2]
    a_pos = a[..., :p]
    a_neg = a[..., p:]

    w_pos = a_pos * a_neg.sum(axis=-1, keepdims=True)
    w_neg = a_neg * a_pos.sum(axis=-1, keepdims=True)

    drift_pos = w_pos @ pos
    drift_neg = w_neg @ neg
    return drift_pos - drift_neg


def drifting_loss(
    gen: jnp.ndarray,
    pos: jnp.ndarray,
    neg: jnp.ndarray,
    temp: float = 0.05,
    exclude_self_neg: bool = False,
    pos_logit_bias=None,
    neg_logit_bias=None,
):
    """Drifting loss: MSE(gen, stopgrad(gen + V))."""
    v = compute_drift(
        gen,
        pos,
        neg,
        temp=temp,
        exclude_self_neg=exclude_self_neg,
        pos_logit_bias=pos_logit_bias,
        neg_logit_bias=neg_logit_bias,
    )
    target = jax.lax.stop_gradient(gen + v)
    return jnp.mean((gen - target) ** 2)


def multi_temp_drifting_loss(
    gen: jnp.ndarray,
    pos: jnp.ndarray,
    neg: jnp.ndarray,
    temps,
    exclude_self_neg: bool = False,
    pos_logit_bias=None,
    neg_logit_bias=None,
):
    """Sum drifting losses computed with multiple temperatures."""
    loss = jnp.asarray(0.0, dtype=gen.dtype)
    for temp in temps:
        loss = loss + drifting_loss(
            gen,
            pos,
            neg,
            temp=float(temp),
            exclude_self_neg=exclude_self_neg,
            pos_logit_bias=pos_logit_bias,
            neg_logit_bias=neg_logit_bias,
        )
    return loss


def _batched_cdist(x, y, eps=1e-12):
    """Compute batched pairwise Euclidean distances."""
    xydot = jnp.einsum('bnd,bmd->bnm', x, y)
    xnorms = jnp.einsum('bnd,bnd->bn', x, x)
    ynorms = jnp.einsum('bmd,bmd->bm', y, y)
    sq_dist = xnorms[:, :, None] + ynorms[:, None, :] - 2.0 * xydot
    return jnp.sqrt(jnp.clip(sq_dist, a_min=eps))


def _bias_to_drift_weight(logit_bias, reference):
    """Convert additive logits to positive, per-state mean-one weights."""
    if logit_bias is None:
        return jnp.ones_like(reference[..., 0])

    bias = jnp.asarray(logit_bias, dtype=reference.dtype)
    if bias.ndim == 3:
        bias = bias[:, 0, :]
    weight = jnp.exp(bias - jnp.max(bias, axis=-1, keepdims=True))
    return weight / jnp.clip(weight.mean(axis=-1, keepdims=True), a_min=1e-8)


def scale_normalized_drifting_loss(
    gen,
    pos,
    neg=None,
    R_list=(0.02, 0.05, 0.2),
    exclude_self_neg=False,
    pos_logit_bias=None,
    neg_logit_bias=None,
):
    """Compute a scale-normalized drift target with balanced multi-radius forces."""
    gen = gen.astype(jnp.float32)
    pos = pos.astype(jnp.float32)
    batch_size, num_gen, action_dim = gen.shape

    # The generated samples are always the primary repulsive set. Current CDP
    # call sites pass gen again as neg with exclude_self_neg=True, so dropping
    # that duplicate avoids counting every generated sample twice.
    if neg is None or exclude_self_neg:
        neg = jnp.zeros_like(gen[:, :0, :])
        neg_logit_bias = None
    else:
        neg = neg.astype(jnp.float32)

    num_neg = neg.shape[1]
    num_pos = pos.shape[1]
    old_gen = jax.lax.stop_gradient(gen)

    weight_gen = jnp.ones((batch_size, num_gen), dtype=gen.dtype)
    weight_neg = _bias_to_drift_weight(neg_logit_bias, neg)
    weight_pos = _bias_to_drift_weight(pos_logit_bias, pos)
    targets = jnp.concatenate([old_gen, neg, pos], axis=1)
    target_weights = jnp.concatenate([weight_gen, weight_neg, weight_pos], axis=1)

    dist = _batched_cdist(old_gen, targets)

    # Estimate scale independently for every state. This prevents states with
    # large action spread from setting the kernel bandwidth for the full batch.
    weighted_dist_sum = (dist * target_weights[:, None, :]).sum(axis=(1, 2))
    weight_mass = num_gen * target_weights.sum(axis=1)
    scale = weighted_dist_sum / jnp.clip(weight_mass, a_min=1e-8)
    scale = jax.lax.stop_gradient(jnp.clip(scale, a_min=1e-3))
    coord_scale = scale[:, None, None] / jnp.sqrt(float(action_dim))

    old_gen_scaled = old_gen / coord_scale
    targets_scaled = targets / coord_scale
    dist_normed = dist / scale[:, None, None]

    diag_mask = jnp.eye(num_gen, dtype=gen.dtype)
    block_mask = jnp.pad(diag_mask, ((0, 0), (0, num_neg + num_pos)))
    dist_normed = dist_normed + block_mask[None, ...] * 100.0

    force_across_R = jnp.zeros_like(old_gen_scaled)
    for R in R_list:
        logits = -dist_normed / float(R)
        affinity_row = jax.nn.softmax(logits, axis=-1)
        affinity_col = jax.nn.softmax(logits, axis=-2)
        affinity = jnp.sqrt(jnp.clip(affinity_row * affinity_col, a_min=1e-6))
        affinity = affinity * target_weights[:, None, :]

        split_idx = num_gen + num_neg
        affinity_neg = affinity[:, :, :split_idx]
        affinity_pos = affinity[:, :, split_idx:]
        coeff_neg = -affinity_neg * affinity_pos.sum(axis=-1, keepdims=True)
        coeff_pos = affinity_pos * affinity_neg.sum(axis=-1, keepdims=True)
        coeff = jnp.concatenate([coeff_neg, coeff_pos], axis=-1)

        force = jnp.einsum('biy,byx->bix', coeff, targets_scaled)
        force = force - coeff.sum(axis=-1)[..., None] * old_gen_scaled

        # Balance radii per state, rather than allowing one bandwidth to
        # dominate because its raw force happens to be larger.
        force_rms = jnp.sqrt(
            jnp.clip(jnp.square(force).mean(axis=(1, 2)), a_min=1e-8)
        )
        force_across_R = force_across_R + force / force_rms[:, None, None]

    goal_scaled = jax.lax.stop_gradient(old_gen_scaled + force_across_R)
    gen_scaled = gen / coord_scale
    return jnp.square(gen_scaled - goal_scaled).mean()


def select_action_subsets(actions: jnp.ndarray, scores: jnp.ndarray, topk: int, bottomk: int):
    """Select top/bottom actions based on per-action scores.

    Args:
        actions: Candidate actions [B, N, A].
        scores: Candidate scores [B, N].
        topk: Number of highest-scoring actions to keep.
        bottomk: Number of lowest-scoring actions to keep.

    Returns:
        top_actions: [B, topk', A]
        bottom_actions: [B, bottomk', A]
        top_scores: [B, topk']
        bottom_scores: [B, bottomk']
    """
    pool_size = actions.shape[1]
    topk = min(max(int(topk), 0), pool_size)
    bottomk = min(max(int(bottomk), 0), pool_size - topk)

    sorted_idx = jnp.argsort(scores, axis=1)
    bottom_idx = sorted_idx[:, :bottomk]
    top_idx = sorted_idx[:, pool_size - topk:] if topk > 0 else sorted_idx[:, :0]

    action_dim = actions.shape[-1]
    top_idx_expanded = jnp.repeat(jnp.expand_dims(top_idx, axis=-1), action_dim, axis=-1)
    bottom_idx_expanded = jnp.repeat(jnp.expand_dims(bottom_idx, axis=-1), action_dim, axis=-1)

    top_actions = jnp.take_along_axis(actions, top_idx_expanded, axis=1)
    bottom_actions = jnp.take_along_axis(actions, bottom_idx_expanded, axis=1)
    top_scores = jnp.take_along_axis(scores, top_idx, axis=1)
    bottom_scores = jnp.take_along_axis(scores, bottom_idx, axis=1)
    return top_actions, bottom_actions, top_scores, bottom_scores


class CDPAgent(flax.struct.PyTreeNode):
    """CDP agent with grouped MPPI positives and behavior-proposal critic bootstrap.

    The implementation is self-contained in this file. It keeps the grouped MPPI
    behavior proposals from the former v5 implementation, skips the single
    highest-Q MPPI positive in actor training, and bootstraps the critic from
    behavior actor proposals.
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def _drifting_loss(
        self,
        gen,
        pos,
        neg,
        temps,
        exclude_self_neg=False,
        pos_logit_bias=None,
        neg_logit_bias=None,
    ):
        """Compute scale-normalized drift loss; subclasses may replace it."""
        return scale_normalized_drifting_loss(
            gen,
            pos,
            neg,
            R_list=tuple(temps),
            exclude_self_neg=exclude_self_neg,
            pos_logit_bias=pos_logit_bias,
            neg_logit_bias=neg_logit_bias,
        )

    def critic_loss(self, batch, grad_params, rng):
        """Compute TD critic loss using behavior-proposal rejection sampling."""
        rng, sample_rng = jax.random.split(rng)
        num_samples = self.config['critic_num_action_samples']

        next_behavior_actions = self._sample_actor_candidates(
            'actor_behavior_onestep_flow',
            batch['next_observations'],
            num_samples,
            sample_rng,
            clip=True,
        )
        batch_size, num_actions, action_dim = next_behavior_actions.shape
        next_obs_repeat = jnp.repeat(batch['next_observations'][:, None, ...], num_actions, axis=1)
        next_obs_flat = next_obs_repeat.reshape(
            (batch_size * num_actions, *batch['next_observations'].shape[1:])
        )
        next_actions_flat = next_behavior_actions.reshape((batch_size * num_actions, action_dim))
        next_behavior_qs = self.network.select('target_critic')(
            next_obs_flat,
            actions=next_actions_flat,
        )
        if self.config['q_agg'] == 'min':
            next_behavior_q = next_behavior_qs.min(axis=0)
        else:
            next_behavior_q = next_behavior_qs.mean(axis=0)
        next_behavior_q = next_behavior_q.reshape((batch_size, num_actions))

        best_idx = jnp.argmax(next_behavior_q, axis=1)
        best_idx = jnp.repeat(best_idx[:, None, None], action_dim, axis=-1)
        next_actions = jnp.take_along_axis(next_behavior_actions, best_idx, axis=1)[:, 0, :]
        next_actions = jax.lax.stop_gradient(jnp.clip(next_actions, -1, 1))

        next_qs = self.network.select('target_critic')(
            batch['next_observations'],
            actions=next_actions,
        )
        if self.config['q_agg'] == 'min':
            next_q = next_qs.min(axis=0)
        else:
            next_q = next_qs.mean(axis=0)

        target_q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_q

        q_data = self.network.select('critic')(batch['observations'], actions=batch['actions'], params=grad_params)
        bellman_loss = jnp.square(q_data - target_q).mean()

        return bellman_loss, {
            'critic_loss': bellman_loss,
            'bellman_loss': bellman_loss,
            'q_data_mean': q_data.mean(),
            'q_data_max': q_data.max(),
            'q_data_min': q_data.min(),
            'next_behavior_q_mean': next_behavior_q.mean(),
            'next_behavior_q_max': next_behavior_q.max(),
            'next_behavior_q_min': next_behavior_q.min(),
            'critic_num_action_samples': jnp.asarray(num_samples),
        }

    def _score_action_candidates(self, observations, candidate_actions):
        """Score a batch of candidate actions with the aggregated critic Q."""
        batch_size, num_actions, action_dim = candidate_actions.shape
        candidate_obs_repeat = jnp.repeat(observations[:, None, ...], num_actions, axis=1)
        candidate_obs_flat = candidate_obs_repeat.reshape(
            (batch_size * num_actions, *observations.shape[1:])
        )
        candidate_actions_flat = candidate_actions.reshape((batch_size * num_actions, action_dim))

        candidate_qs = self.network.select('critic')(
            candidate_obs_flat, actions=candidate_actions_flat
        )
        if self.config['q_agg'] == 'min':
            candidate_q = candidate_qs.min(axis=0)
        else:
            candidate_q = candidate_qs.mean(axis=0)
        candidate_q = candidate_q.reshape((batch_size, num_actions))
        return jax.lax.stop_gradient(candidate_q)

    def _sample_actor_candidates(self, actor_name, observations, num_actions, rng, params=None, clip=False):
        """Sample [B, num_actions, A] candidates from a one-step actor."""
        batch_size = observations.shape[0]
        action_dim = self.config['action_dim']
        noises = jax.random.normal(rng, (batch_size, num_actions, action_dim))
        obs_repeat = jnp.repeat(observations[:, None, ...], num_actions, axis=1)
        obs_flat = obs_repeat.reshape((batch_size * num_actions, *observations.shape[1:]))
        noises_flat = noises.reshape((batch_size * num_actions, action_dim))

        actions_flat = self.network.select(actor_name)(obs_flat, noises_flat, params=params)
        if clip:
            actions_flat = jnp.clip(actions_flat, -1, 1)
        return actions_flat.reshape((batch_size, num_actions, action_dim))

    def _mean_or_zero(self, values, dtype):
        """Mean over a stat tensor, or zero when its static candidate axis is empty."""
        if values.shape[1] > 0:
            return values.mean()
        return jnp.asarray(0.0, dtype=dtype)

    def _policy_positive_bias_and_stats(self, observations, pos_actions, q_dtype, pos_q=None):
        """Build Q-weighted positive logit bias and related logging stats."""
        zero = jnp.asarray(0.0, dtype=q_dtype)
        if pos_actions.shape[1] == 0:
            return None, {
                'pos_q': zero,
                'pos_q_scaled': zero,
                'pos_q_weighted': zero,
                'lam_mean': zero,
                'lam_min': zero,
                'lam_max': zero,
                'pos_entropy': zero,
            }

        if pos_q is None:
            pos_q = self._score_action_candidates(observations, pos_actions)
        else:
            pos_q = jax.lax.stop_gradient(pos_q)
        lam = jax.lax.stop_gradient(1.0 / (jnp.abs(pos_q).mean(axis=1, keepdims=True) + 1e-8))
        pos_q_scaled = pos_q * lam
        pos_probs = jax.nn.softmax(pos_q_scaled / self.config['pos_prob_temp'], axis=1)
        pos_probs = jax.lax.stop_gradient(pos_probs)
        pos_logit_bias = jnp.expand_dims(jnp.log(pos_probs + 1e-12), axis=1)

        return pos_logit_bias, {
            'pos_q': pos_q.mean(),
            'pos_q_scaled': pos_q_scaled.mean(),
            'pos_q_weighted': (pos_probs * pos_q).sum(axis=1).mean(),
            'lam_mean': lam.mean(),
            'lam_min': lam.min(),
            'lam_max': lam.max(),
            'pos_entropy': (-pos_probs * jnp.log(pos_probs + 1e-12)).sum(axis=1).mean(),
        }

    def _mppi_refine_behavior_actions(self, observations, behavior_candidate_actions, grad_params, rng):
        """Refine behavior anchors with anchor-wise MPPI updates."""
        del grad_params
        refined_means = jax.lax.stop_gradient(behavior_candidate_actions)
        q_dtype = behavior_candidate_actions.dtype
        weight_entropy = jnp.asarray(0.0, dtype=q_dtype)
        std_mean = jnp.asarray(0.0, dtype=q_dtype)

        if self.config['mppi_enable']:
            batch_size, num_anchors, action_dim = behavior_candidate_actions.shape
            group_size = self.config['mppi_group_size']
            std = jnp.full(
                (batch_size, num_anchors, 1, action_dim),
                self.config['mppi_std'],
                dtype=q_dtype,
            )
            rng, noise_rng = jax.random.split(rng)
            noise = jax.random.normal(
                noise_rng,
                (batch_size, num_anchors, group_size, action_dim),
                dtype=q_dtype,
            )
            grouped_actions = refined_means[:, :, None, :] + std * noise
            grouped_actions = jnp.clip(grouped_actions, -1.0, 1.0)
            grouped_actions = grouped_actions.at[:, :, 0, :].set(refined_means)
            grouped_actions = jax.lax.stop_gradient(grouped_actions)

            for _ in range(int(self.config['mppi_iters'])):
                rng, sample_rng = jax.random.split(rng)
                flat_actions = grouped_actions.reshape((batch_size, num_anchors * group_size, action_dim))
                candidate_q = self._score_action_candidates(observations, flat_actions)
                candidate_q = candidate_q.reshape((batch_size, num_anchors, group_size))

                q = (candidate_q - candidate_q.mean(axis=2, keepdims=True)) / (
                    candidate_q.std(axis=2, keepdims=True) + 1e-6
                )
                weights = jax.nn.softmax(q / self.config['mppi_temp'], axis=2)
                weights = jax.lax.stop_gradient(weights)

                mean = jnp.sum(weights[..., None] * grouped_actions, axis=2)
                mean = jax.lax.stop_gradient(mean)

                centered_actions = grouped_actions - mean[:, :, None, :]
                var = jnp.sum(weights[..., None] * jnp.square(centered_actions), axis=2)
                std = jnp.sqrt(jnp.maximum(var, 1e-8))
                std = jnp.clip(
                    jnp.maximum(std, self.config['mppi_std_min']),
                    self.config['mppi_std_min'],
                    self.config['mppi_std_max'],
                )
                std = jax.lax.stop_gradient(std)

                refined_means = mean
                refined_means = jnp.clip(refined_means, -1.0, 1.0)
                refined_means = jax.lax.stop_gradient(refined_means)

                noise = jax.random.normal(
                    sample_rng,
                    (batch_size, num_anchors, group_size, action_dim),
                    dtype=q_dtype,
                )
                grouped_actions = refined_means[:, :, None, :] + std[:, :, None, :] * noise
                grouped_actions = jnp.clip(grouped_actions, -1.0, 1.0)
                grouped_actions = grouped_actions.at[:, :, 0, :].set(refined_means)
                grouped_actions = jax.lax.stop_gradient(grouped_actions)

                weight_entropy = (-weights * jnp.log(weights + 1e-12)).sum(axis=2).mean()
                weight_entropy = jax.lax.stop_gradient(weight_entropy)
                std_mean = std.mean()
                std_mean = jax.lax.stop_gradient(std_mean)

        refined_q = self._score_action_candidates(observations, refined_means)
        return refined_means, refined_q, weight_entropy, std_mean

    def _select_after_best(self, actions, scores, topk):
        """Select top-k actions after dropping the single highest-Q candidate."""
        pool_size = actions.shape[1]
        topk = min(max(int(topk), 0), max(pool_size - 1, 0))
        sorted_idx = jnp.argsort(scores, axis=1)
        selected_idx = sorted_idx[:, pool_size - 1 - topk: pool_size - 1]

        action_dim = actions.shape[-1]
        selected_idx_expanded = jnp.repeat(jnp.expand_dims(selected_idx, axis=-1), action_dim, axis=-1)
        selected_actions = jnp.take_along_axis(actions, selected_idx_expanded, axis=1)
        selected_scores = jnp.take_along_axis(scores, selected_idx, axis=1)
        return selected_actions, selected_scores

    def actor_loss(self, batch, grad_params, rng):
        """Train actors with MPPI drift, but skip the highest-Q MPPI positive."""
        behavior_num_neg = self.config['behavior_num_neg']
        policy_num_neg = self.config['policy_num_neg']
        behavior_pool_size = self.config['mppi_num_anchors']
        behavior_bottomk_neg = self.config['behavior_bottomk_neg']
        drift_temps = self.config['drift_temps']
        mppi_num_pos = self.config['mppi_num_pos']
        rng, behavior_noise_rng, policy_x_noise_rng, behavior_pool_noise_rng, mse_rng, mppi_rng = (
            jax.random.split(rng, 6)
        )

        # =========================
        # 1) Behavior actor loss:
        # positives = dataset action only (one positive)
        # =========================
        raw_behavior_actions = self._sample_actor_candidates(
            'actor_behavior_onestep_flow',
            batch['observations'],
            behavior_num_neg,
            behavior_noise_rng,
            params=grad_params,
        )
        behavior_pos_actions = batch['actions'][:, None, :]
        behavior_drift_loss = self._drifting_loss(
            raw_behavior_actions,
            behavior_pos_actions,
            raw_behavior_actions,
            temps=drift_temps,
            exclude_self_neg=True,
        )

        # =========================
        # 2) Policy actor loss:
        # positives = MPPI-refined behavior top-Q
        # negatives = policy self + behavior low-Q
        # =========================
        raw_policy_actions = self._sample_actor_candidates(
            'actor_onestep_flow',
            batch['observations'],
            policy_num_neg,
            policy_x_noise_rng,
            params=grad_params,
        )

        # Build behavior anchors, refine each anchor with grouped MPPI, then
        # use high-Q refined actions as positives while skipping the best one.
        behavior_pool_actions = self._sample_actor_candidates(
            'actor_behavior_onestep_flow',
            batch['observations'],
            behavior_pool_size,
            behavior_pool_noise_rng,
            clip=True,
        )

        behavior_pool_q = self._score_action_candidates(batch['observations'], behavior_pool_actions)
        refined_behavior_actions, refined_behavior_q, mppi_weight_entropy, mppi_std_mean = (
            self._mppi_refine_behavior_actions(
                batch['observations'],
                behavior_pool_actions,
                None,
                mppi_rng,
            )
        )

        if self.config['mppi_keep_behavior_proposals']:
            positive_pool_actions = jnp.concatenate([behavior_pool_actions, refined_behavior_actions], axis=1)
            positive_pool_q = jnp.concatenate([behavior_pool_q, refined_behavior_q], axis=1)
        else:
            positive_pool_actions = refined_behavior_actions
            positive_pool_q = refined_behavior_q

        pos_behavior_actions, pos_behavior_q = self._select_after_best(
            positive_pool_actions,
            positive_pool_q,
            mppi_num_pos,
        )
        _, neg_behavior_actions, _, neg_behavior_q = select_action_subsets(
            behavior_pool_actions,
            behavior_pool_q,
            topk=0,
            bottomk=behavior_bottomk_neg,
        )

        batch_pos_actions = batch['actions'][:, None, :]
        batch_pos_q = self._score_action_candidates(batch['observations'], batch_pos_actions)
        pos_actions = jnp.concatenate([batch_pos_actions, pos_behavior_actions], axis=1)
        pos_actions_q = jnp.concatenate([batch_pos_q, pos_behavior_q], axis=1)
        # neg_actions = jnp.concatenate([raw_policy_actions, neg_behavior_actions], axis=1)
        neg_actions = raw_policy_actions

        if self.config['use_pos_q_bias']:
            pos_logit_bias, pos_stats = self._policy_positive_bias_and_stats(
                batch['observations'],
                pos_actions,
                behavior_pool_q.dtype,
                pos_q=pos_actions_q,
            )
            batch_action_log_weight = jnp.log(
                jnp.asarray(self.config['batch_action_pos_weight'], dtype=behavior_pool_q.dtype)
                + 1e-12
            )
            batch_action_bias = jnp.zeros_like(pos_logit_bias).at[:, :, 0].set(batch_action_log_weight)
            pos_logit_bias = pos_logit_bias + batch_action_bias
        else:
            pos_logit_bias = None
            zero = jnp.asarray(0.0, dtype=behavior_pool_q.dtype)
            pos_stats = {
                'pos_q': self._mean_or_zero(pos_actions_q, behavior_pool_q.dtype),
                'pos_q_scaled': zero,
                'pos_q_weighted': zero,
                'lam_mean': zero,
                'lam_min': zero,
                'lam_max': zero,
                'pos_entropy': zero,
            }

        policy_drift_loss = self._drifting_loss(
            raw_policy_actions,
            pos_actions,
            neg_actions,
            temps=drift_temps,
            exclude_self_neg=True,
            pos_logit_bias=pos_logit_bias,
        )

        # Total actor loss: train both actors together.
        actor_loss = (behavior_drift_loss + policy_drift_loss)

        pos_behavior_q_mean = self._mean_or_zero(pos_behavior_q, behavior_pool_q.dtype)
        neg_behavior_q_mean = self._mean_or_zero(neg_behavior_q, behavior_pool_q.dtype)
        behavior_q_gap = pos_behavior_q_mean - neg_behavior_q_mean

        # MSE is measured on policy actor outputs.
        actions = self.sample_actions(batch['observations'], seed=mse_rng)
        mse = jnp.mean((actions - batch['actions']) ** 2)

        return actor_loss, {
            'actor_loss': actor_loss,
            'behavior_drift_loss': behavior_drift_loss,
            'policy_drift_loss': policy_drift_loss,
            'behavior_pool_q_mean': behavior_pool_q.mean(),
            'behavior_pool_q_max': behavior_pool_q.max(),
            'behavior_pool_q_min': behavior_pool_q.min(),
            'mppi_pos_q_mean': pos_behavior_q_mean,
            'mppi_pos_q_max': pos_behavior_q.max() if pos_behavior_q.shape[1] > 0 else jnp.asarray(0.0, dtype=behavior_pool_q.dtype),
            'mppi_weight_entropy': mppi_weight_entropy,
            'mppi_std_mean': mppi_std_mean,
            'mppi_positive_pool_q_mean': positive_pool_q.mean(),
            'mppi_skipped_best_q_mean': positive_pool_q.max(axis=1).mean(),
            'data_action_q_mean': batch_pos_q.mean(),
            'batch_action_pos_weight': jnp.asarray(self.config['batch_action_pos_weight']),
            'pos_actions_q_mean': pos_stats['pos_q'],
            'pos_q_scaled': pos_stats['pos_q_scaled'],
            'pos_q_weighted': pos_stats['pos_q_weighted'],
            'lam_mean': pos_stats['lam_mean'],
            'lam_min': pos_stats['lam_min'],
            'lam_max': pos_stats['lam_max'],
            'pos_entropy': pos_stats['pos_entropy'],
            'pos_behavior_q_mean': pos_behavior_q_mean,
            'neg_behavior_q_mean': neg_behavior_q_mean,
            'behavior_q_gap': behavior_q_gap,
            'drift_temp_count': jnp.asarray(len(drift_temps)),
            'behavior_num_neg': jnp.asarray(behavior_num_neg),
            'behavior_pool_size': jnp.asarray(behavior_pool_size),
            'mppi_positive_pool_size': jnp.asarray(positive_pool_actions.shape[1]),
            'behavior_topk_pos': jnp.asarray(
                min(max(int(mppi_num_pos), 0), max(positive_pool_actions.shape[1] - 1, 0))
            ),
            'policy_pos_total': jnp.asarray(pos_actions.shape[1]),
            'policy_neg_total': jnp.asarray(neg_actions.shape[1]),
            'policy_num_neg': jnp.asarray(policy_num_neg),
            'mse': mse,
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
            network.params[f'modules_{module_name}'],
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
        """Sample actions from the policy one-step actor."""
        seed, action_seed = jax.random.split(seed)
        noises = jax.random.normal(
            action_seed,
            (
                *observations.shape[: -len(self.config['ob_dims'])],
                self.config['action_dim'],
            ),
        )
        actions = self.network.select('actor_onestep_flow')(observations, noises)
        actions = jnp.clip(actions, -1, 1)
        return actions

    @jax.jit
    def sample_behavior_actions(
        self,
        observations,
        seed=None,
        temperature=1.0,
    ):
        """Sample actions from the behavior one-step actor."""
        del temperature
        seed, action_seed = jax.random.split(seed)
        noises = jax.random.normal(
            action_seed,
            (
                *observations.shape[: -len(self.config['ob_dims'])],
                self.config['action_dim'],
            ),
        )
        actions = self.network.select('actor_behavior_onestep_flow')(observations, noises)
        actions = jnp.clip(actions, -1, 1)
        return actions

    @classmethod
    def create(
        cls,
        seed,
        example_batch,
        config,
    ):
        """Create a new agent.

        Args:
            seed: Random seed.
            example_batch: Example batch.
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_observations = example_batch['observations']
        ex_actions = example_batch['actions']
        ob_dims = ex_observations.shape[1:]
        action_dim = ex_actions.shape[-1]

        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['critic'] = encoder_module()
            encoders['actor_onestep_flow'] = encoder_module()
            encoders['actor_behavior_onestep_flow'] = encoder_module()

        critic_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=2,
            encoder=encoders.get('critic'),
        )
        actor_onestep_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor_onestep_flow'),
        )
        actor_behavior_onestep_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor_behavior_onestep_flow'),
        )

        network_info = dict(
            critic=(critic_def, (ex_observations, ex_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, ex_actions)),
            actor_onestep_flow=(actor_onestep_flow_def, (ex_observations, ex_actions)),
            actor_behavior_onestep_flow=(actor_behavior_onestep_flow_def, (ex_observations, ex_actions)),
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
    config = ml_collections.ConfigDict(
        dict(
            agent_name='cdp',  # Agent name.
            ob_dims=ml_collections.config_dict.placeholder(list),  # Observation dimensions (will be set automatically).
            action_dim=ml_collections.config_dict.placeholder(int),  # Action dimension (will be set automatically).
            lr=3e-4,  # Learning rate.
            batch_size=256,  # Batch size.
            actor_hidden_dims=(512, 512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512, 512),  # Value network hidden dimensions.
            layer_norm=True,  # Whether to use layer normalization.
            actor_layer_norm=False,  # Whether to use layer normalization for the actor.
            discount=0.99,  # Discount factor.
            tau=0.005,  # Target network update rate.
            q_agg='min',  # Aggregation method for target Q values.\
            drift_temps=(0.05, 0.2, 1.0),  # Multi-temperature drift loss (summed over all temps).
            behavior_num_neg=8,  # Number of generated negatives for behavior drifting.
            behavior_bottomk_neg=4,  # Number of low-Q behavior candidates used as policy negatives.
            mppi_enable=True,  # Whether to refine behavior proposals before selecting policy positives.
            mppi_iters=4,  # Number of anchor-wise CEM-MPPI refinement iterations.
            mppi_num_anchors=8,  # Number of behavior anchors / independent MPPI groups.
            mppi_group_size=8,  # Number of perturbation samples around each anchor per MPPI iteration.
            mppi_num_pos=2,  # Select the top-2 refined anchors as policy positives after all MPPI iterations.
            mppi_keep_behavior_proposals=False,  # Use only refined anchors when selecting final positives.
            mppi_temp=1.0,  # Temperature for MPPI proposal reweighting.
            mppi_std=0.10,  # Base/floor std for MPPI resampling.
            mppi_std_min=0.05,  # Minimum diagonal std for MPPI resampling.
            mppi_std_max=0.30,  # Maximum diagonal std for MPPI resampling.
            use_pos_q_bias=True,  # Bias policy drifting positives toward higher-Q selected targets.
            batch_action_pos_weight=1.0,  # Extra multiplicative positive weight for dataset actions.
            policy_num_neg=8,  # Number of policy self-generated negatives for policy drifting.
            pos_prob_temp=10,  # Temperature for exp(Q) sampling probabilities on mixed policy positives.
            critic_num_action_samples=64,  # Number of behavior proposals for critic bootstrap.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
        )
    )
    return config
