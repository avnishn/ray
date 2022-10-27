import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument(
    "--run", type=str, default="PPO", help="The RLlib-registered algorithm to use."
)
parser.add_argument("--num-cpus", type=int, default=0)
parser.add_argument(
    "--framework",
    choices=["tf", "tf2", "torch"],
    default="tf",
    help="The DL framework specifier.",
)
parser.add_argument("--num-workers", type=int, default=0)
parser.add_argument(
    "--use-n-prev-actions",
    type=int,
    default=0,
    help="How many of the previous actions to use as attention input.",
)
parser.add_argument(
    "--use-n-prev-rewards",
    type=int,
    default=0,
    help="How many of the previous rewards to use as attention input.",
)
parser.add_argument("--stop-iters", type=int, default=9999)
parser.add_argument("--stop-timesteps", type=int, default=100000000)
parser.add_argument("--stop-reward", type=float, default=1000.0)

if __name__ == "__main__":
    import ray
    from ray import air, tune

    args = parser.parse_args()

    ray.init(num_cpus=args.num_cpus or None)

    config = {
        "env": "VizdoomBasic-v0",
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "model": {
            "conv_filters": [],
            "use_attention": True,
            "attention_num_transformer_units": 1,
            "attention_dim": 64,
            "attention_num_heads": 2,
            "attention_memory_inference": 100,
            "attention_memory_training": 50,
            "vf_share_layers": True,
            "attention_use_n_prev_actions": args.use_n_prev_actions,
            "attention_use_n_prev_rewards": args.use_n_prev_rewards,
        },
        "framework": args.framework,
        # Run with tracing enabled for tf2.
        "eager_tracing": args.framework == "tf2",
        "num_workers": args.num_workers,
        "vf_loss_coeff": 0.01,
    }

    stop = {
        "training_iteration": args.stop_iters,
        "timesteps_total": args.stop_timesteps,
        "episode_reward_mean": args.stop_reward,
    }

    results = tune.Tuner(
        args.run,
        param_space=config,
        run_config=air.RunConfig(
            stop=stop,
            verbose=2,
            checkpoint_config=air.CheckpointConfig(
                checkpoint_frequency=5,
                checkpoint_at_end=True,
            ),
        ),
    )
    print(results)
    ray.shutdown()
