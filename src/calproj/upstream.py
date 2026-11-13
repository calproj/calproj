"""Locate the MemFLoRA release and put it on ``sys.path``.

The release imports itself as ``src`` (``from src.adapters import ...``) and
its runner as ``experiments``; neither is installed as a package, so its root
directory has to be on ``sys.path``. This repository's own ``src/`` directory
is not a package (it has no ``__init__.py``), so it does not shadow theirs.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

#: the commit of https://github.com/mehmetemreakbulut/MemFLoRA this code was written against
UPSTREAM_COMMIT = "1efcf1ff0c55a7ddcda07e3b1b684452e1bcbd7d"

#: paper names of the two upstream methods, as the release's README maps them
MEMFLORA = "bnpa_fa_postbn_bnr_off_scaled"
MEMFLORA_ORTHOGONAL = "bnpa_fa_postbn_bnr_off_scaled_orthinit"
MEMFLORA_SG = "bnpa_q3_sg"


def upstream_root() -> Path:
    """The directory holding the release: ``$MEMFLORA_ROOT`` or ``third_party/memflora``."""
    env = os.environ.get("MEMFLORA_ROOT")
    root = Path(env) if env else Path(__file__).resolve().parents[2] / "third_party" / "memflora"
    if not (root / "src" / "methods.py").is_file():
        raise FileNotFoundError(
            f"no MemFLoRA release at {root}: set MEMFLORA_ROOT, or fetch commit "
            f"{UPSTREAM_COMMIT[:7]} of github.com/mehmetemreakbulut/MemFLoRA there"
        )
    return root


def enable() -> Path:
    """Make ``import src...`` and ``import experiments...`` resolve to the release."""
    root = upstream_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root
