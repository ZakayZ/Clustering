"""COLA / UrQMD event extraction and dataset loading."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .generate_urqmd_nucleon_dataset import load_dataset_pickle

import colapy


def extract_nucleons_numpy(particles: list[Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert COLA particles to numpy arrays.

    ``pos`` is ``(N, 4)``: ``(t, x, y, z)`` with ``t`` in fm/c and ``x,y,z`` in fm. ``mom`` is ``(E, px, py, pz)`` in MeV/c.
    """
    pos, mom, is_proton = [], [], []
    for p in particles:
        if p.pdg_code == 2212:
            mom.append([p.momentum.e, p.momentum.x, p.momentum.y, p.momentum.z])
            pos.append([p.position.t, p.position.x, p.position.y, p.position.z])
            is_proton.append(True)
        elif p.pdg_code == 2112:
            mom.append([p.momentum.e, p.momentum.x, p.momentum.y, p.momentum.z])
            pos.append([p.position.t, p.position.x, p.position.y, p.position.z])
            is_proton.append(False)
    if not pos:
        return np.zeros((0, 4), np.float64), np.zeros((0, 4), np.float64), np.zeros((0,), bool)
    return np.asarray(pos, np.float64), np.asarray(mom, np.float64), np.asarray(is_proton, bool)


def try_make_urqmd_event_generator() -> Callable[[], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """COLA / UrQMD: one event → ``(pos, mom, is_proton)`` numpy."""

    class W(colapy.WriterBase):
        events: list[Any] = []

        def __init__(self, **kwargs):
            self.events.clear()

        def __call__(self, event_data):
            self.events.append(event_data)

    config = """
<?xml version="1.0" encoding="UTF-8" ?>
<program>
    <generator name="URQMDGenerator"
        pro="197 79"
        tar="197 79"
        nev="1"
        imp="5."
        elb="100."
        tim="200 200"
        generated_config_file="input_file"/>
    <writer name="PythonWriter" class="W"/>
</program>
"""

    def gen_one() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete_on_close=False) as tmp:
            tmp.write(config)
            tmp.close()
            rm = colapy.RunManager().load_module("COLA-Py").load_module("COLA_UrQMD").load_config(tmp.name)
            rm.run(1)
            Path("input_file").unlink(missing_ok=True)
        if not W.events:
            return np.zeros((0, 4)), np.zeros((0, 4)), np.zeros((0,), bool)
        ev = W.events[-1]
        return extract_nucleons_numpy(ev.particles)

    return gen_one


def load_valid_events_from_pkl(pkl: Path) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Load valid events from generated dataset pickle."""
    bundle = load_dataset_pickle(pkl)
    return [(e["pos"], e["mom"], e["is_proton"]) for e in bundle["events"] if e]
