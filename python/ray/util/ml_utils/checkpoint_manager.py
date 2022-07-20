import copy
import enum
import gc
import heapq
import logging
import numbers
import os
import shutil
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import ray
from ray.air import Checkpoint
from ray.tune.result import NODE_IP
from ray.util import log_once
from ray.util.annotations import Deprecated, DeveloperAPI, PublicAPI
from ray.util.ml_utils.util import is_nan

MAX = "max"
MIN = "min"

logger = logging.getLogger(__name__)


@DeveloperAPI
class CheckpointStorage(enum.Enum):
    MEMORY = enum.auto()
    PERSISTENT = enum.auto()


class _TrackedCheckpoint:
    """Checkpoint tracked by a checkpoint manager.

    This class is used to track checkpoints generated by trainables and trainers in
    order to add metadata (e.g. the result, or the node where it has been created)
    and for bookkeeping purposes.

    The data can be an object, a checkpoint directory, or a future to either. Because
    we can't know if it's data or a directory from a future, this class expects
    a ``storage_mode`` that makes the data type explicit.

    The passed metrics can be used to compare performance of different checkpoints.
    The ``checkpoint_id`` is passed as an alternative to be able to order
    checkpoints in time.

    Args:
        dir_or_data: Checkpoint directory, checkpoint data, or a future to either.
        storage_mode: Either MEMORY or PERSISTENT.
        checkpoint_id: Checkpoint number. Will be used to determine checkpoint order
            if metrics are not available. Usually this should be monotonically
            increasing for each tracked checkpoint.
        metrics: Observed metrics for this checkpoint. This is used to determine
            the value of the ``checkpoint_score_attr``.
        node_ip: IP of the node where the checkpoint was generated. Defaults
            to the current node.
    """

    def __init__(
        self,
        dir_or_data: Optional[Union[str, Path, Dict, ray.ObjectRef]],
        storage_mode: CheckpointStorage,
        checkpoint_id: Optional[int] = None,
        metrics: Optional[Dict] = None,
        node_ip: Optional[str] = None,
    ):
        self.dir_or_data = dir_or_data
        self.id = checkpoint_id
        self.storage_mode = storage_mode

        self.metrics = metrics or {}
        self.node_ip = node_ip or self.metrics.get(NODE_IP, None)

        if (
            dir_or_data is not None
            and storage_mode == CheckpointStorage.MEMORY
            and not isinstance(dir_or_data, (dict, ray.ObjectRef))
        ):
            raise ValueError(
                f"Memory checkpoints only support Ray object references and dicts "
                f"as their data. Got: {dir_or_data}"
            )

    def commit(self, path: Optional[Path] = None) -> None:
        """Commit checkpoint to disk, if needed.

        Args:
            path: Path to commit checkpoint to.
        """
        if self.storage_mode == CheckpointStorage.MEMORY:
            # Do not persist memory checkpoints
            return

        if not path:
            # If no path is given, skip
            return

        if not isinstance(self.dir_or_data, dict):
            # Only persist dictionaries
            return

        checkpoint = Checkpoint.from_dict(self.dir_or_data)
        self.dir_or_data = checkpoint.to_directory(str(path))

    def delete(
        self, delete_fn: Optional[Callable[["_TrackedCheckpoint"], None]] = None
    ) -> None:
        """Delete checkpoint from disk, if needed.

        Args:
            delete_fn: Function to be called with the tracked checkpoint as an
                argument. Defaults to removing the local directory/file.
        """
        delete_fn = delete_fn or _default_delete_fn
        try:
            delete_fn(self)
        except Exception as e:
            logger.warning(f"Checkpoint deletion failed: {e}")

    def to_air_checkpoint(self) -> Optional[Checkpoint]:
        from ray.tune.trainable.util import TrainableUtil

        checkpoint_data = self.dir_or_data

        if not checkpoint_data:
            return None

        if isinstance(checkpoint_data, ray.ObjectRef):
            checkpoint_data = ray.get(checkpoint_data)

        if isinstance(checkpoint_data, str):
            try:
                checkpoint_dir = TrainableUtil.find_checkpoint_dir(checkpoint_data)
            except FileNotFoundError:
                if log_once("checkpoint_not_available"):
                    logger.error(
                        f"The requested checkpoint is not available on this node, "
                        f"most likely because you are using Ray client or disabled "
                        f"checkpoint synchronization. To avoid this, enable checkpoint "
                        f"synchronization to cloud storage by specifying a "
                        f"`SyncConfig`. The checkpoint may be available on a different "
                        f"node - please check this location on worker nodes: "
                        f"{checkpoint_data}"
                    )
                return None
            checkpoint = Checkpoint.from_directory(checkpoint_dir)
        elif isinstance(checkpoint_data, bytes):
            checkpoint = Checkpoint.from_bytes(checkpoint_data)
        elif isinstance(checkpoint_data, dict):
            checkpoint = Checkpoint.from_dict(checkpoint_data)
        else:
            raise RuntimeError(f"Unknown checkpoint data type: {type(checkpoint_data)}")

        return checkpoint

    def __repr__(self):
        if self.storage_mode == CheckpointStorage.MEMORY:
            return f"<_TrackedCheckpoint storage='MEMORY' result={self.metrics}>"

        return (
            f"<_TrackedCheckpoint storage='PERSISTENT' "
            f"dir_or_data={self.dir_or_data}>"
        )


def _default_delete_fn(checkpoint: _TrackedCheckpoint):
    if checkpoint.storage_mode != CheckpointStorage.PERSISTENT:
        return

    if isinstance(checkpoint.dir_or_data, (str, bytes, os.PathLike)):
        if os.path.isfile(checkpoint.dir_or_data):
            os.remove(checkpoint.dir_or_data)
            return
        elif os.path.isdir(checkpoint.dir_or_data):
            shutil.rmtree(checkpoint.dir_or_data)
            return
    raise RuntimeError(
        f"Could not delete checkpoint {checkpoint} from disk as it is "
        f"neither file not directory. Path: {checkpoint.dir_or_data}."
    )


class _HeapCheckpointWrapper:
    def __init__(self, priority: Any, tracked_checkpoint: _TrackedCheckpoint):
        self.priority = priority
        self.tracked_checkpoint = tracked_checkpoint

    def __lt__(self, other):
        return self.priority < other.priority

    def __repr__(self):
        return f"_HeapCheckpoint({repr(self.tracked_checkpoint)})"


# Move to ray.air.config when ml_utils is deprecated.
# Doing it now causes a circular import.
@dataclass
@PublicAPI(stability="alpha")
class CheckpointConfig:
    """Configurable parameters for defining the checkpointing strategy.

    Default behavior is to persist all checkpoints to disk. If
    ``num_to_keep`` is set, the default retention policy is to keep the
    checkpoints with maximum timestamp, i.e. the most recent checkpoints.

    Args:
        num_to_keep: The number of checkpoints to keep
            on disk for this run. If a checkpoint is persisted to disk after
            there are already this many checkpoints, then an existing
            checkpoint will be deleted. If this is ``None`` then checkpoints
            will not be deleted. If this is ``0`` then no checkpoints will be
            persisted to disk.
        checkpoint_score_attribute: The attribute that will be used to
            score checkpoints to determine which checkpoints should be kept
            on disk when there are greater than ``num_to_keep`` checkpoints.
            This attribute must be a key from the checkpoint
            dictionary which has a numerical value. Per default, the last
            checkpoints will be kept.
        checkpoint_score_order: Either "max" or "min".
            If "max", then checkpoints with highest values of
            ``checkpoint_score_attribute`` will be kept.
            If "min", then checkpoints with lowest values of
            ``checkpoint_score_attribute`` will be kept.
        checkpoint_frequency: Number of iterations between checkpoints. If 0
            this will disable checkpointing.
            Please note that most trainers will still save one checkpoint at
            the end of training.
            This attribute is only supported
            by trainers that don't take in custom training loops.
        checkpoint_at_end: If True, will save a checkpoint at the end of training.
            This attribute is only supported by trainers that don't take in
            custom training loops. Defaults to True for trainers that support it
            and False for generic function trainables.

    """

    num_to_keep: Optional[int] = None
    checkpoint_score_attribute: Optional[str] = None
    checkpoint_score_order: str = MAX
    checkpoint_frequency: int = 0
    checkpoint_at_end: Optional[bool] = None

    def __post_init__(self):
        if self.num_to_keep is not None and self.num_to_keep < 0:
            raise ValueError(
                f"Received invalid num_to_keep: "
                f"{self.num_to_keep}. "
                f"Must be None or non-negative integer."
            )
        if self.checkpoint_score_order not in (MAX, MIN):
            raise ValueError(
                f"checkpoint_score_order must be either " f'"{MAX}" or "{MIN}".'
            )

        if self.checkpoint_frequency < 0:
            raise ValueError(
                f"checkpoint_frequency must be >=0, got {self.checkpoint_frequency}"
            )

    @property
    def _tune_legacy_checkpoint_score_attr(self) -> Optional[str]:
        """Same as ``checkpoint_score_attr`` in ``tune.run``.

        Only used for Legacy API compatibility.
        """
        if self.checkpoint_score_attribute is None:
            return self.checkpoint_score_attribute
        prefix = ""
        if self.checkpoint_score_order == MIN:
            prefix = "min-"
        return f"{prefix}{self.checkpoint_score_attribute}"


# Alias for backwards compatibility

deprecation_message = (
    "`CheckpointStrategy` is deprecated and will be removed in "
    "the future. Please use `ray.air.config.CheckpointStrategy` "
    "instead."
)


@Deprecated(message=deprecation_message)
@dataclass
class CheckpointStrategy(CheckpointConfig):
    def __post_init__(self):
        warnings.warn(deprecation_message)
        super().__post_init__()


class _CheckpointManager:
    """Common checkpoint management and bookkeeping class for Ray Train and Tune.

    This class acts as the common core for checkpoint bookkeeping in Ray ML libraries.
    On a high level, this manager keeps a reference to all stored checkpoints
    (both in-memory and on-disk checkpoints). For on-disk checkpoints, it
    keeps a configured number of checkpoints according to specified metrics.

    The manager supports lazy data writing by utilizing the
    ``_TrackedCheckpoint.commit()`` API, which is only invoked if the checkpoint
    should be persisted to disk.

    Args:
        checkpoint_strategy: Checkpoint strategy defining how many and which
            checkpoints to keep.
        latest_checkpoint_id: First checkpoint ID to use (e.g. in case we
            continue training an existing experiment).
        delete_fn: Function that takes a TrackedCheckpoint and deletes it from disk
            or memory upon request.

    """

    # If memory checkpoints should be persisted
    _persist_memory_checkpoints: bool = False

    def __init__(
        self,
        checkpoint_strategy: CheckpointConfig,
        latest_checkpoint_id: int = 0,
        delete_fn: Optional[Callable[["_TrackedCheckpoint"], None]] = None,
    ):
        self._checkpoint_strategy = checkpoint_strategy or CheckpointConfig()

        # Incremental unique checkpoint ID of this run.
        self._latest_checkpoint_id = latest_checkpoint_id

        # Used for keeping top K checkpoints.
        self._top_persisted_checkpoints: List[_HeapCheckpointWrapper] = []

        # Best checkpoint altogether.
        # Used for exposing best_checkpoint_path.
        self._best_persisted_checkpoint: Optional[_TrackedCheckpoint] = None
        self._latest_persisted_checkpoint: Optional[_TrackedCheckpoint] = None
        self._latest_memory_checkpoint: Optional[_TrackedCheckpoint] = None

        # Deletion of some checkpoints should be deferred. Specifically, if the
        # latest persisted checkpoint should be deleted, we will only delete it
        # once a new checkpoint came in (so that `_latest_persisted_checkpoint` is
        # always available).
        self._checkpoints_to_clean_up = set()

        self._delete_fn = delete_fn

    def set_delete_fn(
        self, delete_fn: Optional[Callable[["_TrackedCheckpoint"], None]]
    ):
        """Update the function called to delete persisted checkpoints.

        Args:
            delete_fn: Function that takes a tracked checkpoint as an argument and
                deletes it from disk.
        """
        self._delete_fn = delete_fn

    def register_checkpoint(self, checkpoint: _TrackedCheckpoint):
        """Register new checkpoint and add to bookkeeping.

        This method will register a new checkpoint and add it to the internal
        bookkeeping logic. This means the checkpoint manager will decide if
        this checkpoint should be kept, and if older or worse performing
        checkpoints should be deleted.

        Args:
            checkpoint: Tracked checkpoint object to add to bookkeeping.
        """
        checkpoint.id = checkpoint.id or self._latest_checkpoint_id

        if checkpoint.storage_mode == CheckpointStorage.MEMORY:
            self._replace_latest_memory_checkpoint(checkpoint)

            if self._persist_memory_checkpoints:
                persisted_checkpoint = copy.copy(checkpoint)
                persisted_checkpoint.storage_mode = CheckpointStorage.PERSISTENT
            else:
                persisted_checkpoint = None
        else:
            persisted_checkpoint = checkpoint

        if persisted_checkpoint and self._checkpoint_strategy.num_to_keep != 0:
            self._process_persistent_checkpoint(persisted_checkpoint)

        self._latest_checkpoint_id += 1

    def _replace_latest_memory_checkpoint(self, memory_checkpoint: _TrackedCheckpoint):
        assert memory_checkpoint.storage_mode == CheckpointStorage.MEMORY
        self._latest_memory_checkpoint = memory_checkpoint
        # Avoid memory leaks on k8s pods
        gc.collect()

    def _replace_latest_persisted_checkpoint(
        self, persisted_checkpoint: _TrackedCheckpoint
    ):
        second_to_latest_persisted_checkpoint = self._latest_persisted_checkpoint
        self._latest_persisted_checkpoint = persisted_checkpoint

        if self._checkpoint_strategy.num_to_keep == 0:
            self._maybe_delete_persisted_checkpoint(
                second_to_latest_persisted_checkpoint
            )

    def _maybe_replace_best_persisted_checkpoint(
        self, persisted_checkpoint: _TrackedCheckpoint
    ):
        if self._best_persisted_checkpoint is None:
            self._best_persisted_checkpoint = persisted_checkpoint
        else:
            old_score = self._get_checkpoint_score(self._best_persisted_checkpoint)
            candidate_score = self._get_checkpoint_score(persisted_checkpoint)
            if candidate_score >= old_score:
                self._best_persisted_checkpoint = persisted_checkpoint

    def _get_checkpoint_score(
        self, checkpoint: _TrackedCheckpoint
    ) -> Tuple[bool, numbers.Number, int]:
        checkpoint_score_attribute = (
            self._checkpoint_strategy.checkpoint_score_attribute
        )
        if checkpoint_score_attribute not in checkpoint.metrics:
            logger.error(
                f"Result dict has no key: {checkpoint_score_attribute}. "
                f"checkpoint_score_attr must be set to a key in the "
                f"result dict. Valid keys are: {list(checkpoint.metrics.keys())}"
            )
            checkpoint_result = float("-inf")
        else:
            checkpoint_result = checkpoint.metrics[checkpoint_score_attribute]

        checkpoint_score_order = self._checkpoint_strategy.checkpoint_score_order
        if checkpoint_score_order == MAX:
            order_factor = 1.0
        else:
            order_factor = -1.0

        checkpoint_score = order_factor * checkpoint_result

        if not isinstance(checkpoint_score, numbers.Number):
            raise ValueError(
                f"Unable to persist checkpoint for "
                f"checkpoint_score_attribute: "
                f"{checkpoint_score_attribute} with value "
                f"{checkpoint_score}. "
                f"This attribute must be numerical."
            )

        return (
            not is_nan(checkpoint_score),
            checkpoint_score if not is_nan(checkpoint_score) else 0,
            checkpoint.id,
        )

    def _process_persistent_checkpoint(self, checkpoint: _TrackedCheckpoint):
        assert checkpoint.storage_mode == CheckpointStorage.PERSISTENT

        checkpoint_score = self._get_checkpoint_score(checkpoint)
        wrapped_checkpoint = _HeapCheckpointWrapper(
            priority=checkpoint_score, tracked_checkpoint=checkpoint
        )

        if self._checkpoint_strategy.num_to_keep is None:
            # Keep all checkpoints
            checkpoint.commit(path=self._get_next_checkpoint_path())
            self._replace_latest_persisted_checkpoint(checkpoint)
            self._top_persisted_checkpoints.append(wrapped_checkpoint)
        elif (
            len(self._top_persisted_checkpoints) < self._checkpoint_strategy.num_to_keep
        ):
            # Heap is not full yet, so keep this checkpoint
            checkpoint.commit(path=self._get_next_checkpoint_path())
            heapq.heappush(self._top_persisted_checkpoints, wrapped_checkpoint)
            self._replace_latest_persisted_checkpoint(checkpoint)
        elif wrapped_checkpoint.priority >= self._top_persisted_checkpoints[0].priority:
            # Priority is higher than current worst checkpoint, so replace worst
            checkpoint.commit(path=self._get_next_checkpoint_path())
            worst_checkpoint = heapq.heappushpop(
                self._top_persisted_checkpoints, wrapped_checkpoint
            ).tracked_checkpoint

            # Only remove if checkpoint data is different
            if worst_checkpoint.dir_or_data != checkpoint.dir_or_data:
                self._maybe_delete_persisted_checkpoint(worst_checkpoint)
                logger.debug(f"Removed worst checkpoint from " f"{worst_checkpoint}.")

            self._replace_latest_persisted_checkpoint(checkpoint)
        else:
            # If the latest checkpoint has the same or lower priority, skip it.
            self._skip_persisted_checkpoint(checkpoint)

        self._maybe_replace_best_persisted_checkpoint(persisted_checkpoint=checkpoint)
        self._cleanup_checkpoints()

    def _maybe_delete_persisted_checkpoint(
        self, persisted_checkpoint: _TrackedCheckpoint
    ):
        if persisted_checkpoint == self._latest_persisted_checkpoint:
            self._checkpoints_to_clean_up.add(persisted_checkpoint)
        else:
            self._delete_persisted_checkpoint(persisted_checkpoint=persisted_checkpoint)

    def _delete_persisted_checkpoint(self, persisted_checkpoint: _TrackedCheckpoint):
        persisted_checkpoint.delete(delete_fn=self._delete_fn)
        self._checkpoints_to_clean_up.discard(persisted_checkpoint)

    def _cleanup_checkpoints(self):
        for checkpoint in list(self._checkpoints_to_clean_up):
            self._maybe_delete_persisted_checkpoint(persisted_checkpoint=checkpoint)

    def _skip_persisted_checkpoint(self, persisted_checkpoint: _TrackedCheckpoint):
        logger.debug(f"Skipping checkpoint due to low score: {persisted_checkpoint}.")
        self._checkpoints_to_clean_up.add(persisted_checkpoint)

    def _get_next_checkpoint_path(self) -> Optional[Path]:
        return None

    def __del__(self):
        self._cleanup_checkpoints()

    def __getstate__(self):
        state = self.__dict__.copy()

        # Do not serialize the delete fn
        state.pop("_delete_fn", None)

        # Avoid serializing the memory checkpoint.
        state["_newest_memory_checkpoint"] = _TrackedCheckpoint(
            dir_or_data=None,
            checkpoint_id=0,
            storage_mode=CheckpointStorage.MEMORY,
        )
        return state

    def __setstate__(self, state):
        state["_delete_fn"] = None
        self.__dict__.update(state)
