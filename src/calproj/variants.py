"""The projections compared: where the subspace comes from, and how its rows are scaled.

Every variant is the MemFLoRA adapter of the release with a different frozen
down-projection ``P``; nothing else changes. ``gauss`` is MemFLoRA itself and
``sg`` is MemFLoRA-SG.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import upstream

#: where the subspace of the projection comes from
SUBSPACES = (
    "upstream",            # whatever the release draws: Gaussian or random orthogonal
    "signs",          # i.i.d. signs: the Gaussian's second moments without a multiplication
    "select-random",       # r input channels picked at random: a channel-sparse weight update
    "select-top",          # the r input channels with the most energy
    "activation-source",   # leading eigenvectors of the input moment on source data (offline)
    "activation-target",   # the same on the target calibration batch (on device, forward-only)
    "activation-target-stream",  # the same through the two-pass O(C r) estimator
    "gradient-target",     # leading input-side directions of the sketched weight gradient (one replay)
)
#: how the rows are scaled
SCALES = (
    "upstream",   # as drawn: N(0, 1/r) entries, +-1/sqrt(r) signs, or orthonormal rows
    "unit",       # the Gaussian draw divided by sqrt(C/r): rows of unit norm in expectation
    "norm",       # unit-norm rows times sqrt(C/r), the expected row norm of N(0, 1/r)
    "balanced",   # every bottleneck channel carries tr(Sigma)/r, as N(0, 1/r) does in expectation
    "energy",     # one factor for all rows: the bottleneck carries tr(Sigma) in total
)
#: the data the balanced and energy scales (and the channel ranking) are measured on
REFERENCES = ("auto", "source", "target")


@dataclass(frozen=True)
class Variant:
    name: str
    method: str            # the release's method name
    subspace: str = "upstream"
    scale: str = "upstream"
    sg: bool = False       # train P by forward replay (MemFLoRA-SG)
    label: str = ""        # how tables name it
    nomean: bool = False   # keep the rows orthogonal to the mean input of the target batch
    reference: str = "auto"  # auto: the source for source-calibrated subspaces, else the target batch
    rotate: bool = False   # mix the calibrated directions by a random rotation: the same subspace, another basis
    center: bool = False   # subtract the mean input from the bottleneck (a constant shift of q)
    bias: bool = False     # train a per-channel bias beside U (its gradient needs no saved input)

    def __post_init__(self) -> None:
        if self.subspace not in SUBSPACES or self.scale not in SCALES or self.reference not in REFERENCES:
            raise ValueError(f"unknown subspace, scale or reference in variant {self.name}")
        if self.scale == "upstream" and self.subspace not in ("upstream", "signs"):
            raise ValueError(f"variant {self.name}: only a drawn projection keeps its own scale")
        if self.scale == "unit" and self.subspace != "upstream":
            raise ValueError(f"variant {self.name}: 'unit' rescales the release's Gaussian draw")
        if self.scale == "energy" and not self.subspace.startswith(("activation", "gradient")):
            raise ValueError(f"variant {self.name}: 'energy' is a scale of calibrated subspaces")
        if self.rotate and not self.subspace.startswith(("activation", "gradient")):
            raise ValueError(f"variant {self.name}: only a calibrated subspace is rotated")
        if self.nomean and self.scale not in ("balanced", "upstream"):
            raise ValueError(f"variant {self.name}: 'nomean' rows keep their scale or are rebalanced")
        if self.center and (self.nomean or self.scale != "balanced"):
            raise ValueError(f"variant {self.name}: a centred bottleneck is rebalanced, and keeps the mean direction")

    @property
    def uses_source(self) -> bool:
        """Whether the variant reads statistics of the source data (computed offline)."""
        if self.reference == "source":
            return True
        return self.reference == "auto" and self.subspace == "activation-source"

    @property
    def data_free(self) -> bool:
        """Whether the projection is built without looking at any activation."""
        return self.scale in ("upstream", "unit", "norm") and not self.subspace.startswith(("activation", "gradient")) \
            and self.subspace != "select-top" and not self.nomean and not self.center


_M, _O, _S = upstream.MEMFLORA, upstream.MEMFLORA_ORTHOGONAL, upstream.MEMFLORA_SG

VARIANTS: dict[str, Variant] = {
    v.name: v
    for v in (
        # the release's projections
        Variant("gauss", _M, label="Gaussian (MemFLoRA)"),
        Variant("orth", _O, label="Random orthogonal"),
        Variant("orth-norm", _O, scale="norm", label="Random orthogonal, norm-matched"),
        Variant("sg", _S, sg=True, label="Gaussian, trained by replay (MemFLoRA-SG)"),
        # calibrated projections
        Variant("pca-src", _M, "activation-source", "balanced", label="Calibrated on source activations"),
        Variant("pca-tgt", _M, "activation-target", "balanced", label="Calibrated on target activations"),
        Variant("grad", _M, "gradient-target", "balanced", label="Calibrated on a target gradient sketch"),
        # ablations of the calibrated projection
        Variant("pca-src-energy", _M, "activation-source", "energy", label="Source activations, unbalanced"),
        Variant("pca-src-norm", _M, "activation-source", "norm", label="Source activations, norm-matched"),
        Variant("pca-tgt-stream", _M, "activation-target-stream", "balanced",
                label="Target activations, streaming estimator"),
        Variant("pca-src-sg", _S, "activation-source", "balanced", sg=True,
                label="Source activations, then trained by replay"),
        # the calibrated subspace in a random basis: Adam acts per coordinate, so the basis matters
        Variant("pca-src-rot", _M, "activation-source", "balanced", rotate=True,
                label="Source activations, randomly rotated basis"),
        Variant("pca-tgt-rot", _M, "activation-target", "balanced", rotate=True,
                label="Target activations, randomly rotated basis"),
        # the scale, apart from the subspace
        Variant("gauss-unit", _M, scale="unit", label="Gaussian, unit-norm rows"),
        # projections that cost nothing to store or to apply
        Variant("signs", _M, "signs", label="Random signs"),
        Variant("select-norm", _M, "select-random", "norm", label="Random channels, norm-matched"),
        Variant("select", _M, "select-random", "balanced", label="Random channels, balanced on the target batch"),
        Variant("select-src", _M, "select-random", "balanced", reference="source",
                label="Random channels, balanced on source data"),
        Variant("select-top", _M, "select-top", "balanced", label="Top-energy channels of the target batch"),
        Variant("select-top-src", _M, "select-top", "balanced", reference="source",
                label="Top-energy channels of the source data"),
        # diagnostics: the same bottleneck energy, with or without the mean input direction
        Variant("gauss-balanced", _M, scale="balanced", label="Gaussian, balanced on the target batch"),
        Variant("gauss-nomean", _M, scale="balanced", nomean=True, label="Gaussian, mean direction removed"),
        Variant("gauss-nomean-keep", _M, nomean=True, label="Gaussian, mean direction removed, scale kept"),
        Variant("gauss-nomean-src", _M, scale="balanced", nomean=True, reference="source",
                label="Gaussian, source mean direction removed"),
        Variant("cal", _M, scale="balanced", center=True, label="Calibrated on the target batch"),
        Variant("cal-src", _M, scale="balanced", center=True, reference="source",
                label="Calibrated on source data"),
        Variant("gauss-bias", _M, bias=True, label="Gaussian, with a trainable bias"),
        Variant("gauss-nomean-bias", _M, scale="balanced", nomean=True, bias=True,
                label="Gaussian, mean direction removed, with a bias"),
        Variant("cal-bias", _M, scale="balanced", center=True, bias=True,
                label="Calibrated, with a trainable bias"),
        Variant("signs-cal", _M, "signs", "balanced", center=True, label="Random signs, calibrated"),
        Variant("select-cal", _M, "select-random", "balanced", center=True,
                label="Random channels, calibrated"),
        Variant("select-cal-bias", _M, "select-random", "balanced", center=True, bias=True,
                label="Random channels, calibrated, with a bias"),
        Variant("sg-cal", _S, scale="balanced", center=True, sg=True,
                label="Calibrated, then trained by replay"),
        Variant("sg-nomean", _S, scale="balanced", nomean=True, sg=True,
                label="Gaussian, mean direction removed, then trained by replay"),
        Variant("pca-tgt-nomean", _M, "activation-target", "balanced", nomean=True,
                label="Target activations, mean direction removed"),
    )
}


def get(name: str) -> Variant:
    try:
        return VARIANTS[name]
    except KeyError:
        raise KeyError(f"unknown variant {name!r}; known: {', '.join(VARIANTS)}") from None
