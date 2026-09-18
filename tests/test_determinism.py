import hashlib
import pathlib

import numpy as np
import pytest

from sim.run import run

SCENES = ["plaza", "hub", "corridor"]


def _sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


@pytest.mark.parametrize("scene", SCENES)
def test_same_seed_same_path_byte_identical(tmp_path, scene):
    a = run(scene, 200, seed=7, frames=90, out=str(tmp_path / "a.csv"))
    b = run(scene, 200, seed=7, frames=90, out=str(tmp_path / "b.csv"))
    assert _sha(a) == _sha(b)


def test_different_seed_differs(tmp_path):
    a = run("plaza", 200, seed=1, frames=90, out=str(tmp_path / "a.csv"))
    b = run("plaza", 200, seed=2, frames=90, out=str(tmp_path / "b.csv"))
    assert _sha(a) != _sha(b)


@pytest.mark.parametrize("scene", SCENES)
def test_tier_histograms_sum_to_agent_count(tmp_path, scene):
    p = run(scene, 200, seed=3, frames=60, out=str(tmp_path / "a.csv"))
    rows = np.loadtxt(p, delimiter=",", skiprows=1, usecols=range(5, 13), dtype=np.int64)
    assert (rows[:, :4].sum(1) == 200).all() and (rows[:, 4:].sum(1) == 200).all()
