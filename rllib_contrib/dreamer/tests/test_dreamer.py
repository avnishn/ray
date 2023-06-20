import unittest

from gymnasium.spaces import Box
from rllib_dreamer.dreamer import DreamerConfig

import ray
from ray.rllib.examples.env.random_env import RandomEnv
from ray.rllib.utils.test_utils import framework_iterator


class TestDreamer(unittest.TestCase):
    """Sanity tests for Dreamer."""

    def setUp(self):
        ray.init()

    def tearDown(self):
        ray.shutdown()

    def test_dreamer_compilation(self):
        """Test whether Dreamer can be built with all frameworks."""
        config = DreamerConfig()
        config.environment(
            env=RandomEnv,
            env_config={
                "observation_space": Box(-1.0, 1.0, (3, 64, 64)),
                "action_space": Box(-1.0, 1.0, (3,)),
            },
        )
        # Num episode chunks per batch.
        # Length (ts) of an episode chunk in a batch.
        # Sub-iterations per .train() call.
        config.training(batch_size=2, batch_length=20, dreamer_train_iters=4)

        num_iterations = 1

        # Test against all frameworks.
        for _ in framework_iterator(config, frameworks="torch"):
            algo = config.build()
            for i in range(num_iterations):
                results = algo.train()
                print(results)
            # check_compute_single_action(trainer, include_state=True)
            algo.stop()


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main(["-v", __file__]))
