import sys
import gc
import copy
import time
import queue
import pickle
import shutil
import threading
from pathlib import Path
import torch
from lerobot.utils.utils import log_say


def _get_dataset_writer(dataset):
    return getattr(dataset, "writer", None)


def _get_episode_buffer(dataset):
    try:
        return dataset.episode_buffer
    except AttributeError:
        pass

    writer = _get_dataset_writer(dataset)
    if writer is not None and hasattr(writer, "episode_buffer"):
        return writer.episode_buffer
    raise RuntimeError("Unable to access dataset episode buffer for async save.")


def _set_episode_buffer(dataset, episode_buffer):
    updated = False
    writer = _get_dataset_writer(dataset)
    if writer is not None and hasattr(writer, "episode_buffer"):
        writer.episode_buffer = episode_buffer
        updated = True

    try:
        getattr(dataset, "episode_buffer")
    except AttributeError:
        pass
    else:
        try:
            dataset.episode_buffer = episode_buffer
            updated = True
        except AttributeError:
            pass

    if not updated:
        raise RuntimeError("Unable to replace dataset episode buffer for async save.")


def _get_image_writer(dataset):
    """0.6.1: image writer lives on the DatasetWriter; keep the old dataset-level fallback."""
    writer = _get_dataset_writer(dataset)
    image_writer = getattr(writer, "image_writer", None) if writer is not None else None
    if image_writer is None:  # compatibility with older dataset layouts
        image_writer = getattr(dataset, "image_writer", None)
    return image_writer


def _to_int(value):
    if isinstance(value, (list, tuple)):
        return int(value[0])
    if hasattr(value, "item"):
        try:
            return int(value.item())
        except (TypeError, ValueError):
            pass
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(value[0])


def _episode_buffer_size(episode_buffer):
    return _to_int(episode_buffer.get("size", 0))


def _episode_buffer_index(episode_buffer):
    return _to_int(episode_buffer["episode_index"])


def _current_episode_index(dataset):
    try:
        return _episode_buffer_index(_get_episode_buffer(dataset))
    except Exception:
        return dataset.num_episodes


def _create_empty_episode_buffer(dataset, episode_index, template_episode_buffer):
    writer = _get_dataset_writer(dataset)

    if writer is not None and hasattr(writer, "_create_episode_buffer"):
        episode_buffer = writer._create_episode_buffer()
    elif hasattr(dataset, "create_episode_buffer"):
        episode_buffer = dataset.create_episode_buffer(episode_index=episode_index)
    elif hasattr(dataset, "_create_episode_buffer"):
        episode_buffer = dataset._create_episode_buffer()
    else:
        episode_buffer = copy.deepcopy(template_episode_buffer)
        for key, value in list(episode_buffer.items()):
            if key == "size":
                episode_buffer[key] = 0
            elif key == "episode_index":
                continue
            elif isinstance(value, list):
                episode_buffer[key] = []
            else:
                episode_buffer[key] = []

    if _episode_buffer_index(episode_buffer) != episode_index:
        episode_buffer["episode_index"] = episode_index
    return episode_buffer


def _create_next_episode_buffer(dataset, current_episode_buffer):
    current_episode_index = _episode_buffer_index(current_episode_buffer)
    return _create_empty_episode_buffer(dataset, current_episode_index + 1, current_episode_buffer)


def _release_unused_memory():
    gc.collect()
    if not sys.platform.startswith("linux"):
        return
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


class ImageWriterSaveBarrier:
    def __init__(self, image_writer, max_pending_image_tasks=8192):
        self.image_writer = image_writer
        self._max_pending_image_tasks = max(0, int(max_pending_image_tasks))
        self._condition = threading.Condition()
        self._submitted = 0
        self._completed = 0
        self._reported_backpressure = False
        self._original_task_done = None

    @classmethod
    def install(cls, dataset, max_pending_image_tasks=8192):
        image_writer = _get_image_writer(dataset)
        if image_writer is None:
            return None

        existing = getattr(image_writer, "_uf_save_barrier", None)
        if existing is not None:
            return existing

        barrier = cls(image_writer, max_pending_image_tasks)
        if barrier._install():
            image_writer._uf_save_barrier = barrier
            return barrier
        return None

    def _install(self):
        if getattr(self.image_writer, "num_processes", 0) != 0:
            return False

        task_queue = getattr(self.image_writer, "queue", None)
        if task_queue is None:
            return False

        self._original_task_done = task_queue.task_done

        def save_image(image, fpath, compress_level=1):
            if isinstance(image, torch.Tensor):
                image = image.cpu().numpy()
            with self._condition:
                while (
                    self._max_pending_image_tasks > 0
                    and self._submitted - self._completed >= self._max_pending_image_tasks
                ):
                    if not self._reported_backpressure:
                        log_say(
                            f"[Async] waiting for image writer backlog "
                            f"({self._max_pending_image_tasks}) to release memory...",
                            False
                        )
                        self._reported_backpressure = True
                    self._condition.wait()
                self._submitted += 1
            task_queue.put((image, fpath, compress_level))

        def task_done():
            self._original_task_done()
            with self._condition:
                self._completed += 1
                self._condition.notify_all()

        self.image_writer.save_image = save_image
        task_queue.task_done = task_done
        return True

    def snapshot(self):
        with self._condition:
            return self._submitted

    def wait_until_done(self, snapshot):
        with self._condition:
            while self._completed < snapshot:
                self._condition.wait()


class AsyncEpisodeSaver:
    _STOP = object()

    def __init__(self, dataset, max_pending_saves=5, max_pending_image_tasks=8192):
        self.dataset = dataset
        self._queue = queue.Queue()
        self._max_pending_saves = max(1, int(max_pending_saves))
        self._max_pending_image_tasks = max(0, int(max_pending_image_tasks))
        self._pending_save_slots = threading.BoundedSemaphore(self._max_pending_saves)
        self._spool_dir = Path(dataset.root) / ".uf_async_episode_spool"
        self._spool_dir.mkdir(parents=True, exist_ok=True)
        self._total_cnts = 0
        self._finish_cnts = 0
        self._exception = None
        self._wait_context = threading.local()
        self._image_writer_barrier = ImageWriterSaveBarrier.install(dataset, self._max_pending_image_tasks)
        if self._image_writer_barrier is not None:
            self._patch_dataset_image_writer_wait()
        elif _get_image_writer(dataset) is not None:
            log_say("[Async] image writer barrier is unavailable; async saves may wait for active recording.", False)
        self._thread = threading.Thread(target=self._run, name="uf-async-episode-saver", daemon=True)
        self._thread.start()

    def submit_current_episode(self):
        self._raise_if_failed()
        if not self._pending_save_slots.acquire(blocking=False):
            log_say(f"[Async] waiting for pending saves ({self._max_pending_saves}) to release memory...", False)
            self._pending_save_slots.acquire()

        try:
            self._raise_if_failed()
            episode_buffer = _get_episode_buffer(self.dataset)
            if _episode_buffer_size(episode_buffer) == 0:
                raise RuntimeError("Cannot async save an empty episode buffer.")

            episode_index = _episode_buffer_index(episode_buffer)
            next_episode_buffer = _create_next_episode_buffer(self.dataset, episode_buffer)
            _set_episode_buffer(self.dataset, next_episode_buffer)
            image_writer_snapshot = self._image_writer_snapshot()
            spool_path = self._write_episode_spool(episode_index, episode_buffer)
            self._queue.put((episode_index, episode_buffer, image_writer_snapshot, spool_path))
            log_say(f"[Async] episode {episode_index} submitted for async save, all_pending={self._queue.qsize()}", False)
        except BaseException:
            self._pending_save_slots.release()
            raise
        return episode_index

    def wait_idle(self):
        self._queue.join()
        self._raise_if_failed()

    def close(self):
        try:
            self._queue.join()
            self._queue.put(self._STOP)
            self._queue.join()
            self._thread.join()
            self._raise_if_failed()
        finally:
            self._restore_dataset_image_writer_wait()

    def _run(self):
        while True:
            item = self._queue.get()
            release_pending_slot = False
            episode_buffer = None
            episode_index = None   # 提前初始化
            try:
                if item is self._STOP:
                    return
                release_pending_slot = True
                episode_index, episode_buffer, image_writer_snapshot, spool_path = item
                log_say(f"[Episode {episode_index}] async saving ...", False)
                try:
                    self._wait_context.image_writer_snapshot = image_writer_snapshot
                    self.dataset.save_episode(episode_data=episode_buffer)
                except TypeError as exc:
                    if "episode_data" in str(exc):
                        raise RuntimeError(
                            "--async-save requires LeRobotDataset.save_episode(episode_data=...)."
                        ) from exc
                    raise
                self._delete_saved_image_dirs(episode_index)
                self._delete_episode_spool(spool_path)
                log_say(f"[Episode {episode_index}] async save finish, all_pending={self._queue.qsize()}", False)
            except BaseException as exc:
                self._exception = exc
                label = episode_index if episode_index is not None else "unknown"
                log_say(f"[Episode {label}] async save failed, {exc}", False)
            finally:
                if hasattr(self._wait_context, "image_writer_snapshot"):
                    del self._wait_context.image_writer_snapshot
                if isinstance(episode_buffer, dict):
                    episode_buffer.clear()
                item = None
                episode_buffer = None
                _release_unused_memory()
                if release_pending_slot:
                    self._pending_save_slots.release()
                self._queue.task_done()

    def _write_episode_spool(self, episode_index, episode_buffer):
        spool_path = self._spool_dir / f"episode-{episode_index:06d}.pkl"
        tmp_path = spool_path.with_suffix(".pkl.tmp")
        with tmp_path.open("wb") as f:
            pickle.dump(
                {"episode_index": episode_index, "episode_buffer": episode_buffer},
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        tmp_path.replace(spool_path)
        return spool_path

    def _delete_episode_spool(self, spool_path):
        if spool_path is not None:
            Path(spool_path).unlink(missing_ok=True)

    def _patch_dataset_image_writer_wait(self):
        # 0.6.1: writer.save_episode() calls self._wait_image_writer() before encoding;
        # patch it on the writer so the async thread only waits for the snapshot taken
        # at submit time, without blocking the recording loop.
        writer = _get_dataset_writer(self.dataset)
        if writer is None:
            return
        self._original_wait_image_writer = getattr(writer, "_wait_image_writer", None)
        if self._original_wait_image_writer is None:
            return

        def wait_image_writer_for_current_context():
            image_writer_snapshot = getattr(self._wait_context, "image_writer_snapshot", None)
            if image_writer_snapshot is not None:
                self._image_writer_barrier.wait_until_done(image_writer_snapshot)
            else:
                self._original_wait_image_writer()

        writer._wait_image_writer = wait_image_writer_for_current_context
        self._patched_writer = writer

    def _restore_dataset_image_writer_wait(self):
        writer = getattr(self, "_patched_writer", None)
        if writer is None:
            writer = _get_dataset_writer(self.dataset)
        if writer is not None and self._original_wait_image_writer is not None:
            writer._wait_image_writer = self._original_wait_image_writer

    def _image_writer_snapshot(self):
        if self._image_writer_barrier is None:
            return None
        return self._image_writer_barrier.snapshot()

    def _delete_saved_image_dirs(self, episode_index):
        writer = _get_dataset_writer(self.dataset)
        meta = getattr(writer, "_meta", getattr(self.dataset, "meta", None))
        image_keys = getattr(meta, "image_keys", [])
        image_dir_owner = writer if writer is not None and hasattr(writer, "_get_image_file_dir") else self.dataset
        if not image_keys or not hasattr(image_dir_owner, "_get_image_file_dir"):
            return

        for cam_key in image_keys:
            img_dir = image_dir_owner._get_image_file_dir(episode_index, cam_key)
            if img_dir.is_dir():
                shutil.rmtree(img_dir)

    def _raise_if_failed(self):
        if self._exception is not None:
            raise RuntimeError("Async episode save failed.") from self._exception
