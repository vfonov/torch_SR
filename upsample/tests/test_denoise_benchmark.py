"""Cross-backend runtime comparison for one NLM denoising pass.

Exercises ``scripts/benchmark_denoise.run_all`` on a single small volume with
a single shared uniform sigma, so every available backend -- ``c`` (CFFI,
CPU), ``pytorch/cpu``, ``pytorch/cuda`` (when present) and the torch-free
``numpy`` -- runs the exact same ``regularize()`` call.  Backends that are
not available are reported as skipped, not failed.

This is a lightweight guard that the benchmark wiring works and that the
backends still agree where they are meant to; the printed table (visible
with ``pytest -s``) is the actual timing comparison.  Run
``python scripts/benchmark_denoise.py`` for a realistically sized run.
"""

import importlib.util
import pathlib
import sys

import numpy as np

_UPSAMPLE = pathlib.Path(__file__).resolve().parent.parent
if str(_UPSAMPLE) not in sys.path:
    sys.path.insert(0, str(_UPSAMPLE))

_spec = importlib.util.spec_from_file_location(
    "benchmark_denoise", _UPSAMPLE / "scripts" / "benchmark_denoise.py")
assert _spec is not None and _spec.loader is not None
_bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bench)


def test_all_backends_run_and_agree(small_volume, capsys):
    vol = np.ascontiguousarray(
        small_volume.numpy().astype(np.float64) * 256.0)
    sigma = _bench.build_sigma(vol)  # uniform, as denoise.py's default
    factors = (2, 2, 2)

    results = _bench.run_all(vol, sigma, factors, repeat=1)

    with capsys.disabled():
        print()
        print(_bench.format_table(results, vol.shape, factors, repeat=1))
        print()

    ran = {r["label"]: r for r in results if r["time"] is not None}
    assert "numpy" in ran and "pytorch/cpu" in ran

    for label, r in ran.items():
        assert r["time"] > 0.0, label
        assert r["out"].shape == vol.shape, label
        assert np.isfinite(r["out"]).all(), label

    # numpy is a port of the pytorch CPU path -- they must match tightly.
    assert np.allclose(ran["pytorch/cpu"]["out"], ran["numpy"]["out"],
                       rtol=1e-6, atol=1e-8)

    # The C (CFFI) kernel is the project's numerical reference and the
    # benchmark's reference column.  With a real factor and a uniform sigma
    # all backends do the same work and agree on the interior; the v+f border
    # still differs (C skips out-of-bounds neighbours, the others reflect-pad).
    if "c" in ran:
        assert _bench._reference_label(results) == "c"
        ref_c = _bench._interior(ran["c"]["out"])
        for label in ("pytorch/cpu", "numpy"):
            got = _bench._interior(ran[label]["out"])
            assert np.abs(got - ref_c).max() / vol.mean() < 1e-6, label


def test_plain_denoise_factors_skip_c(small_volume):
    """factors=(1,1,1): the C wrapper is the identity, so it is skipped and
    numpy becomes the reference column."""
    vol = np.ascontiguousarray(
        small_volume.numpy().astype(np.float64) * 256.0)
    results = _bench.run_all(vol, _bench.build_sigma(vol), (1, 1, 1), repeat=1)

    by_label = {r["label"]: r for r in results}
    assert by_label["c"]["time"] is None
    assert "identity" in by_label["c"]["why"]
    assert _bench._reference_label(results) == "numpy"
