import gymnasium as gym
import unittest

from ray.rllib.utils.framework import try_import_tf
import ray

from ray.rllib.core.rl_trainer.trainer_runner import TrainerRunner
from ray.rllib.core.testing.tf.bc_module import DiscreteBCTFModule
from ray.rllib.core.testing.tf.bc_rl_trainer import BCTfRLTrainer
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID, MultiAgentBatch
from ray.rllib.utils.test_utils import check, get_cartpole_dataset_reader

tf1, tf, tfv = try_import_tf()
tf1.executing_eagerly()


class TestTrainerRunner(unittest.TestCase):
    """This test is setup for 2 gpus."""

    # TODO: Make a unittest that does not need 2 gpus to run.
    # So that the user can run it locally as well.
    @classmethod
    def setUp(cls) -> None:
        ray.init()

    @classmethod
    def tearDown(cls) -> None:
        ray.shutdown()

    def test_update_multigpu(self):
        """Test training in a 2 gpu setup and that weights are synchronized."""
        env = gym.make("CartPole-v1")
        trainer_class = BCTfRLTrainer
        trainer_cfg = dict(
            module_class=DiscreteBCTFModule,
            module_kwargs={
                "observation_space": env.observation_space,
                "action_space": env.action_space,
                "model_config": {"hidden_dim": 32},
            },
            optimizer_config={"lr": 1e-3},
            in_test=True,
        )
        runner = TrainerRunner(
            trainer_class, trainer_cfg, compute_config=dict(num_gpus=2)
        )

        reader = get_cartpole_dataset_reader(batch_size=500)

        min_loss = float("inf")
        for iter_i in range(1000):
            batch = reader.next()
            results_worker_0, results_worker_1 = runner.update(batch.as_multi_agent())

            loss = (
                results_worker_0["loss"]["total_loss"]
                + results_worker_1["loss"]["total_loss"]
            ) / 2
            min_loss = min(loss, min_loss)
            print(f"[iter = {iter_i}] Loss: {loss:.3f}, Min Loss: {min_loss:.3f}")
            # The loss is initially around 0.69 (ln2). When it gets to around
            # 0.57 the return of the policy gets to around 100.
            if min_loss < 0.57:
                break
            self.assertEqual(
                results_worker_0["mean_weight"]["default_policy"],
                results_worker_1["mean_weight"]["default_policy"],
            )
        self.assertLess(min_loss, 0.57)

    def test_add_remove_module(self):
        env = gym.make("CartPole-v1")
        trainer_class = BCTfRLTrainer
        trainer_cfg = dict(
            module_class=DiscreteBCTFModule,
            module_kwargs={
                "observation_space": env.observation_space,
                "action_space": env.action_space,
                "model_config": {"hidden_dim": 32},
            },
            optimizer_config={"lr": 1e-3},
            in_test=True,
        )
        runner = TrainerRunner(
            trainer_class, trainer_cfg, compute_config=dict(num_gpus=2)
        )

        reader = get_cartpole_dataset_reader(batch_size=500)
        batch = reader.next()

        # update once with the default policy
        results = runner.update(batch.as_multi_agent())
        module_ids_before_add = {DEFAULT_POLICY_ID}
        new_module_id = "test_module"

        # add a test_module
        self.add_module_helper(env, new_module_id, runner)

        # do training that includes the test_module
        results = runner.update(
            MultiAgentBatch(
                {new_module_id: batch, DEFAULT_POLICY_ID: batch}, batch.count
            )
        )

        # check that module weights are updated across workers and synchronized
        for i in range(1, len(results)):
            for module_id in results[i]["mean_weight"].keys():
                assert (
                    results[i]["mean_weight"][module_id]
                    == results[i - 1]["mean_weight"][module_id]
                )

        # check that module ids are updated to include the new module
        module_ids_after_add = {DEFAULT_POLICY_ID, new_module_id}
        for result in results:
            # remove the total_loss key since its not a module key
            self.assertEqual(set(result["loss"]) - {"total_loss"}, module_ids_after_add)

        # remove the test_module
        runner.remove_module(module_id=new_module_id)

        # run training without the test_module
        results = runner.update(batch.as_multi_agent())

        # check that module weights are updated across workers and synchronized
        for i in range(1, len(results)):
            for module_id in results[i]["mean_weight"].keys():
                assert (
                    results[i]["mean_weight"][module_id]
                    == results[i - 1]["mean_weight"][module_id]
                )

        # check that module ids are updated after remove operation to not
        # include the new module
        for result in results:
            # remove the total_loss key since its not a module key
            self.assertEqual(
                set(result["loss"]) - {"total_loss"}, module_ids_before_add
            )

    def test_trainer_runner_no_gpus(self):
        import os

        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        env = gym.make("CartPole-v1")
        trainer_class = BCTfRLTrainer
        trainer_cfg = dict(
            module_class=DiscreteBCTFModule,
            module_kwargs={
                "observation_space": env.observation_space,
                "action_space": env.action_space,
                "model_config": {"hidden_dim": 32},
            },
            optimizer_config={"lr": 1e-3},
            in_test=True,
        )
        runner = TrainerRunner(
            trainer_class, trainer_cfg, compute_config=dict(num_gpus=0)
        )

        local_trainer = trainer_class(**trainer_cfg)
        local_trainer.build()

        # make the state of the trainer and the local runner identical
        local_trainer.set_state(runner.get_state()[0])

        reader = get_cartpole_dataset_reader(batch_size=500)
        batch = reader.next()
        batch = batch.as_multi_agent()
        check(local_trainer.update(batch), runner.update(batch)[0])

        new_module_id = "test_module"

        # add a test_module
        self.add_module_helper(env, new_module_id, runner)
        self.add_module_helper(env, new_module_id, local_trainer)

        # make the state of the trainer and the local runner identical
        local_trainer.set_state(runner.get_state()[0])

        # do another update
        batch = reader.next()
        ma_batch = MultiAgentBatch(
            {new_module_id: batch, DEFAULT_POLICY_ID: batch}, env_steps=batch.count
        )
        check(local_trainer.update(ma_batch), runner.update(ma_batch)[0])

        check(local_trainer.get_state(), runner.get_state()[0])

    def add_module_helper(self, env, module_id, runner_or_trainer):
        runner_or_trainer.add_module(
            module_id=module_id,
            module_cls=DiscreteBCTFModule,
            module_kwargs={
                "observation_space": env.observation_space,
                "action_space": env.action_space,
                "model_config": {"hidden_dim": 32},
            },
            optimizer_cls=tf.keras.optimizers.Adam,
        )


if __name__ == "__main__":
    import pytest
    import sys

    sys.exit(pytest.main(["-v", __file__]))
