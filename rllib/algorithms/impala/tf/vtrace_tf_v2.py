from typing import List, Union
from ray.rllib.utils.framework import try_import_tf

_, tf, _ = try_import_tf()


def make_time_major(
    tensor: Union[tf.Tensor, List[tf.Tensor]],
    *,
    trajectory_len: int = None,
    recurrent_seq_len: int = None,
    drop_last: bool = False,
):
    """Swaps batch and trajectory axis.

    Args:
        tensor: A tensor or list of tensors to swap the axis of.
            NOTE: Each tensor must have the shape [B * T] where B is the batch size and
            T is the trajectory length.
        trajectory_len: The length of each trajectory being transformed.
            If None then `recurrent_seq_len` must be set.
        recurrent_seq_len: Sequence lengths if recurrent.
            If None then `trajectory_len` must be set.
        drop_last: A bool indicating whether to drop the last
            trajectory item.

    Note: Either `trajectory_len` or `recurrent_seq_len` must be set. `trajectory_len`
        should be used in cases where tensor is not produced from a
        RNN/recurrent module. `recurrent_seq_len` should be used in those cases instead.

    Returns:
        A tensor with swapped axes or a list of tensors with swapped axes.
    """
    if isinstance(tensor, list):
        return [
            make_time_major(_tensor, trajectory_len, recurrent_seq_len, drop_last)
            for _tensor in tensor
        ]

    assert (trajectory_len != recurrent_seq_len) and (
        trajectory_len is None or recurrent_seq_len is None
    ), "Either trajectory_len or recurrent_seq_len must be set."

    if recurrent_seq_len:
        B = tf.shape(recurrent_seq_len)[0]
        T = tf.shape(tensor)[0] // B
    else:
        T = trajectory_len
        B = tf.shape(tensor)[0] // T
    rs = tf.reshape(tensor, tf.concat([[B, T], tf.shape(tensor)[1:]], axis=0))

    # swap B and T axes
    res = tf.transpose(rs, [1, 0] + list(range(2, 1 + int(tf.shape(tensor).shape[0]))))

    if drop_last:
        return res[:-1]
    return res


def vtrace_tf2(
    *,
    target_action_log_probs: tf.Tensor,
    behaviour_action_log_probs: tf.Tensor,
    discounts: tf.Tensor,
    rewards: tf.Tensor,
    values: tf.Tensor,
    bootstrap_value: tf.Tensor,
    clip_rho_threshold: Union[float, tf.Tensor] = 1.0,
    clip_pg_rho_threshold: Union[float, tf.Tensor] = 1.0,
):
    r"""V-trace for softmax policies.

    Calculates V-trace actor critic targets for softmax polices as described in

    "IMPALA: Scalable Distributed Deep-RL with
    Importance Weighted Actor-Learner Architectures"
    by Espeholt, Soyer, Munos et al.

    Target policy refers to the policy we are interested in improving and
    behaviour policy refers to the policy that generated the given
    rewards and actions.

    In the notation used throughout documentation and comments, T refers to the
    time dimension ranging from 0 to T-1. B refers to the batch size and
    ACTION_SPACE refers to the list of numbers each representing a number of
    actions.

    Args:
        target_action_log_probs: Action log probs from the target policy. A float32
            tensor of shape [T, B].
        behaviour_action_log_probs: Action log probs from the behaviour policy. A
            float32 tensor of shape [T, B].
        discounts: A float32 tensor of shape [T, B] with the discount encountered when
            following the behaviour policy. This will be 0 for terminal timesteps
            (done=True) and gamma (the discount factor) otherwise.
        rewards: A float32 tensor of shape [T, B] with the rewards generated by
            following the behaviour policy.
        values: A float32 tensor of shape [T, B] with the value function estimates
            wrt. the target policy.
        bootstrap_value: A float32 of shape [B] with the value function estimate at
            time T.
        clip_rho_threshold: A scalar float32 tensor with the clipping threshold for
            importance weights (rho) when calculating the baseline targets (vs).
            rho^bar in the paper.
        clip_pg_rho_threshold: A scalar float32 tensor with the clipping threshold
            on rho_s in \rho_s \delta log \pi(a|x) (r + \gamma v_{s+1} - V(x_s)).
    """
    log_rhos = target_action_log_probs - behaviour_action_log_probs

    discounts = tf.convert_to_tensor(discounts, dtype=tf.float32)
    rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
    values = tf.convert_to_tensor(values, dtype=tf.float32)
    bootstrap_value = tf.convert_to_tensor(bootstrap_value, dtype=tf.float32)
    if clip_rho_threshold is not None:
        clip_rho_threshold = tf.convert_to_tensor(clip_rho_threshold, dtype=tf.float32)
    if clip_pg_rho_threshold is not None:
        clip_pg_rho_threshold = tf.convert_to_tensor(
            clip_pg_rho_threshold, dtype=tf.float32
        )

    # Make sure tensor ranks are consistent.
    rho_rank = log_rhos.shape.ndims  # Usually 2.
    values.shape.assert_has_rank(rho_rank)
    bootstrap_value.shape.assert_has_rank(rho_rank - 1)
    discounts.shape.assert_has_rank(rho_rank)
    rewards.shape.assert_has_rank(rho_rank)
    if clip_rho_threshold is not None:
        clip_rho_threshold.shape.assert_has_rank(0)
    if clip_pg_rho_threshold is not None:
        clip_pg_rho_threshold.shape.assert_has_rank(0)

    rhos = tf.math.exp(log_rhos)
    if clip_rho_threshold is not None:
        clipped_rhos = tf.minimum(clip_rho_threshold, rhos, name="clipped_rhos")
    else:
        clipped_rhos = rhos

    cs = tf.minimum(1.0, rhos, name="cs")
    # Append bootstrapped value to get [v1, ..., v_t+1]
    values_t_plus_1 = tf.concat(
        [values[1:], tf.expand_dims(bootstrap_value, 0)], axis=0
    )

    deltas = clipped_rhos * (rewards + discounts * values_t_plus_1 - values)

    # All sequences are reversed, computation starts from the back.
    sequences = (
        tf.reverse(discounts, axis=[0]),
        tf.reverse(cs, axis=[0]),
        tf.reverse(deltas, axis=[0]),
    )

    # V-trace vs are calculated through a scan from the back to the
    # beginning of the given trajectory.
    def scanfunc(acc, sequence_item):
        discount_t, c_t, delta_t = sequence_item
        return delta_t + discount_t * c_t * acc

    initial_values = tf.zeros_like(bootstrap_value)
    vs_minus_v_xs = tf.nest.map_structure(
        tf.stop_gradient,
        tf.scan(
            fn=scanfunc,
            elems=sequences,
            initializer=initial_values,
            parallel_iterations=1,
            name="scan",
        ),
    )
    # Reverse the results back to original order.
    vs_minus_v_xs = tf.reverse(vs_minus_v_xs, [0])

    # Add V(x_s) to get v_s.
    vs = tf.add(vs_minus_v_xs, values)

    # Advantage for policy gradient.
    vs_t_plus_1 = tf.concat([vs[1:], tf.expand_dims(bootstrap_value, 0)], axis=0)
    if clip_pg_rho_threshold is not None:
        clipped_pg_rhos = tf.minimum(clip_pg_rho_threshold, rhos)
    else:
        clipped_pg_rhos = rhos
    pg_advantages = clipped_pg_rhos * (rewards + discounts * vs_t_plus_1 - values)

    # Make sure no gradients backpropagated through the returned values.
    return tf.stop_gradient(vs), tf.stop_gradient(pg_advantages)
