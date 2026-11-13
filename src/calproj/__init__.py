"""Calibrated projections for memory-floor low-rank adaptation of CNNs.

A memory-floor adapter (MemFLoRA, arXiv:2610.08669) trains an up-projection
``U`` behind a frozen pointwise down-projection ``P``. This package chooses
``P`` from data instead of drawing it at random:

- :mod:`calproj.projection` holds the mathematics: the subspace (the leading
  eigenvectors of a calibration moment) and the scale of the rows;
- :mod:`calproj.moments` collects the moments, from activations or gradients;
- :mod:`calproj.adapt` runs one adaptation with the MemFLoRA release in
  ``third_party/memflora`` (:mod:`calproj.upstream` finds it).
"""

__version__ = "0.1.0"
