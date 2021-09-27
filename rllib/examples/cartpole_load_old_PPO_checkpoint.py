from ray.rllib.agents.ppo import PPOTrainer


if __name__ == "__main__":
    import ray
    from ray import tune
    path = "/Users/avnish/ray_results/torch_086_experiment/PPO_CartPole-v0_0_2021-09-22_12-58-07knlikspn/checkpoint_2/checkpoint-2"
    ray.init(local_mode=True)

    config = {
            "num_sgd_iter": 5,
            "model": {
                "vf_share_layers": True,
            },
            "vf_loss_coeff": 0.0001,
            "env": "CartPole-v0",
            # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
            "framework": 'torch',
        }
    stop_reward = 150.0
    stop_timesteps = 100000
    stop_iters = 2

    stop = {
        "training_iteration": stop_iters,
        "timesteps_total": stop_timesteps,
        "episode_reward_mean": stop_reward,
    }

    trainer = PPOTrainer(config)
    # tensorflow
    # trainer.restore("/Users/avnish/ray_results/PPO/PPO_CartPole-v0_0_2021-09-22_11-52-17wftg2wxn/checkpoint_1/checkpoint-1")

    # pytorch
    trainer.restore(path)
    import ipdb; ipdb.set_trace()

    # tune.run(PPOTrainer, config=config, stop=stop, verbose=2, restore="/Users/avnish/ray_results/torch_086_experiment/PPO_CartPole-v0_0_2021-09-22_12-58-07knlikspn/checkpoint_2/checkpoint-2")
    # To run the Trainer without tune.run, using our LSTM model and
    # manual state-in handling, do the following:

    # Example (use `config` from the above code):
    # >> import numpy as np
    # >> from ray.rllib.agents.ppo import PPOTrainer
    # >>
    # >> trainer = PPOTrainer(config)
    # >> lstm_cell_size = config["model"]["lstm_cell_size"]
    # >> env = StatelessCartPole()
    # >> obs = env.reset()
    # >>
    # >> # range(2) b/c h- and c-states of the LSTM.
    # >> init_state = state = [
    # ..     np.zeros([lstm_cell_size], np.float32) for _ in range(2)
    # .. ]
    # >> prev_a = 0
    # >> prev_r = 0.0
    # >>
    # >> while True:
    # >>     a, state_out, _ = trainer.compute_single_action(
    # ..         obs, state, prev_a, prev_r)
    # >>     obs, reward, done, _ = env.step(a)
    # >>     if done:
    # >>         obs = env.reset()
    # >>         state = init_state
    # >>         prev_a = 0
    # >>         prev_r = 0.0
    # >>     else:
    # >>         state = state_out
    # >>         prev_a = a
    # >>         prev_r = reward

    # results = tune.run(args.run, config=config, stop=stop, verbose=2, restore="/Users/avnish/ray_results/PPO/PPO_StatelessCartPole_0_2021-09-21_14-27-59lnd6b3ks/checkpoint_1/")

    ray.shutdown()
