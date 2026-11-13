"""Run directories: JSON sidecars, provenance, and one record per unit of work.

A run directory holds ``config.json`` (what was asked), ``env.json`` (where and
with what it ran), ``metrics.json`` (every finished unit) and ``units.jsonl``,
the append-only log ``metrics.json`` is rebuilt from. A resubmitted job reads
``units.jsonl`` and skips what is done, so a wall-time kill loses one unit.
"""

from __future__ import annotations

import json
import os
import platform
import socket
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable


def write_json(path: Path, payload: Any) -> None:
    """Write JSON atomically: a killed job leaves the previous file, never half of a new one."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=1, sort_keys=True, allow_nan=True)
            handle.write("\n")
        os.chmod(tmp, 0o644)  # mkstemp creates 0600
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def git_stamp(root: Path) -> dict[str, str]:
    """The commit recorded in ``.git_commit`` by whatever copied the tree to a machine without its ``.git``."""
    stamp = Path(root) / ".git_commit"
    if not stamp.is_file():
        return {}
    pairs = (line.split("=", 1) for line in stamp.read_text().splitlines() if "=" in line)
    return {key.strip(): value.strip() for key, value in pairs}


def env_info(root: Path) -> dict[str, Any]:
    """Library versions, hardware and scheduler context of the current process."""
    import numpy
    import sklearn
    import torch
    import torchvision

    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": numpy.__version__,
        "scikit_learn": sklearn.__version__,
        "torch_threads": torch.get_num_threads(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "host": socket.gethostname(),
        "cluster": os.environ.get("SLURM_CLUSTER_NAME", ""),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "slurm_job_name": os.environ.get("SLURM_JOB_NAME", ""),
        "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK", ""),
        "cuda": torch.cuda.is_available(),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git": git_stamp(root),
    }
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            models = {line.split(":", 1)[1].strip() for line in handle if line.startswith("model name")}
        info["cpu"] = sorted(models)[0] if models else ""
    except OSError:
        info["cpu"] = ""
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["cuda_version"] = torch.version.cuda
    return info


class UnitLog:
    """Finished units of a run, keyed by a tuple the caller chooses."""

    def __init__(self, run_dir: Path, key_fields: Iterable[str]) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.key_fields = tuple(key_fields)
        self.path = self.run_dir / "units.jsonl"
        self.units: list[dict] = []
        self._done: set[tuple] = set()
        if self.path.is_file():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    unit = json.loads(line)
                except json.JSONDecodeError:
                    continue  # the line a kill cut short; its unit is simply run again
                self._remember(unit)

    def key(self, unit: dict) -> tuple:
        return tuple(unit[f] for f in self.key_fields)

    def _remember(self, unit: dict) -> None:
        key = self.key(unit)
        if key not in self._done:
            self._done.add(key)
            self.units.append(unit)

    def done(self, **key) -> bool:
        return tuple(key[f] for f in self.key_fields) in self._done

    def add(self, unit: dict) -> None:
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(unit, sort_keys=True, allow_nan=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self._remember(unit)

    def write_metrics(self, **summary) -> None:
        write_json(self.run_dir / "metrics.json", {**summary, "n_units": len(self.units), "units": self.units})
