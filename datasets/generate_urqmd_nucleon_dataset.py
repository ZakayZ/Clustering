#!/usr/bin/env python3
"""
Generate a dataset of UrQMD events as nucleon (p, n) clouds for training.

All events are written to **one** pickle file (default: ``<out>/dataset.pkl``) with:

- ``events``: length-``n`` list; each entry is ``{"pos", "mom", "is_proton"}`` or ``None`` if that index failed
- ``pos`` (N, 4) as ``(t, x, y, z)`` (fm/c, fm) or (N, 3) spatial fm; ``mom`` (N, 4) MeV as (E, px, py, pz), ``is_proton`` (N,) bool

``meta`` and ``failed`` are included in the same pickle. Load with ``load_dataset_pickle``.

Parallelism: **one worker process per chunk**, **one** ``RunManager``, one ``load_config``,
then ``run(n)`` with ``n = len(chunk)`` (all UrQMD events for that process in one call).
Each chunk gets its own UrQMD ``rsd`` (see ``--urqmd-random-base``) so workers do not
share the same default clock seed.

XML ``nev`` is set to that same ``n``. Each chunk uses a **short-path** temp directory
(built with ``tempfile.mkdtemp``) because UrQMD’s Fortran driver stores ``ftn09`` in
``character*77`` — long paths under ``clustering/`` are truncated and UrQMD fails to
open the input file. Inside that directory: ``in``, ``t.dat``, ``run.xml``.

Example::

  python clustering/generate_urqmd_nucleon_dataset.py --tables path/to/tables.dat \\
      --n 10000 --workers 6 --out clustering/datasets/urqmd_nucleons_10k
"""

import argparse
import importlib.util
import os
import pickle
import secrets
import shutil
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

_WORKER_EVENTS: list[Any] = []

_CLUSTERING_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _CLUSTERING_DIR.parent

_URQMD_RSD_MOD = (1 << 31) - 2  # positive 31-bit UrQMD ``rsd`` values


def _urqmd_chunk_rsd(random_base: int, global_first_index: int) -> int:
    """Distinct positive UrQMD ``rsd`` per chunk; reproducible given ``random_base``."""
    x = (random_base + global_first_index * 1_000_003) % _URQMD_RSD_MOD
    return int(x + 1)


def _extract_nucleons(particles: list[Any]) -> tuple[Any, Any, Any]:
    import numpy as np

    pos, mom, isp = [], [], []
    for p in particles:
        if p.pdg_code == 2212:
            mom.append([p.momentum.e, p.momentum.x, p.momentum.y, p.momentum.z])
            pos.append([p.position.t, p.position.x, p.position.y, p.position.z])
            isp.append(True)
        elif p.pdg_code == 2112:
            mom.append([p.momentum.e, p.momentum.x, p.momentum.y, p.momentum.z])
            pos.append([p.position.t, p.position.x, p.position.y, p.position.z])
            isp.append(False)
    if not pos:
        return (
            np.zeros((0, 4), dtype=np.float64),
            np.zeros((0, 4), dtype=np.float64),
            np.zeros((0,), dtype=bool),
        )
    return (
        np.asarray(pos, dtype=np.float64),
        np.asarray(mom, dtype=np.float64),
        np.asarray(isp, dtype=bool),
    )


def _resolve_tables_src(explicit: Path | None) -> Path:
    if explicit is not None:
        p = explicit.expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"--tables not a file: {p}")
        return p
    for c in (
        _REPO_ROOT / "tests" / "tables.dat",
        _REPO_ROOT / "urqmd-4.0" / "tables.dat",
        _CLUSTERING_DIR / "tables.dat",
    ):
        if c.is_file():
            return c.resolve()
    raise FileNotFoundError(
        "Could not find tables.dat. Pass --tables /path/to/tables.dat "
        "(or place tables.dat under clustering/, urqmd-4.0/, or tests/)."
    )


def _split_indices(n: int, n_workers: int) -> list[list[int]]:
    """Partition ``range(n)`` into at most ``n_workers`` contiguous chunks."""
    if n <= 0:
        return []
    w = max(1, min(n_workers, n))
    size = (n + w - 1) // w
    chunks: list[list[int]] = []
    start = 0
    while start < n:
        chunks.append(list(range(start, min(start + size, n))))
        start += size
    return chunks


def _register_dataset_writer(import_colapy: Any) -> None:
    class DatasetWriter(import_colapy.WriterBase):
        def __init__(self, **kwargs: Any) -> None:
            pass

        def __call__(self, event_data: Any) -> None:
            _WORKER_EVENTS.append(event_data)

    globals()["DatasetWriter"] = DatasetWriter
    m = sys.modules.get("__main__")
    if m is not None:
        setattr(m, "DatasetWriter", DatasetWriter)


def _process_chunk(
    indices: list[int],
    tables_src: str,
    phys: dict[str, str],
    keep_staging: bool,
    staging_parent: str | None,
    chunk_rsd: int,
) -> list[dict[str, Any]]:
    """One process: one ``RunManager``, one ``load_config``, then ``run(n)`` for ``n=len(indices)``."""

    import colapy

    _register_dataset_writer(colapy)

    pro, tar, imp, elb, tim = phys["pro"], phys["tar"], phys["imp"], phys["elb"], phys["tim"]
    n_run = len(indices)
    if n_run == 0:
        return []

    parent = staging_parent if staging_parent else None
    staging_dir = Path(tempfile.mkdtemp(prefix="urq_", dir=parent))
    gen_path = (staging_dir / "in").resolve()
    tab_path = (staging_dir / "t.dat").resolve()
    xml_path = (staging_dir / "run.xml").resolve()

    results: list[dict[str, Any]] = []
    try:
        shutil.copy2(tables_src, tab_path)
        os.environ["URQMD_TAB"] = str(tab_path)

        gen_attr = gen_path.as_posix()
        cfg = f"""<?xml version="1.0" encoding="UTF-8" ?>
<program>
    <generator name="URQMDGenerator"
        pro="{pro}"
        tar="{tar}"
        nev="{n_run}"
        imp="{imp}"
        elb="{elb}"
        tim="{tim}"
        rsd="{chunk_rsd}"
        generated_config_file="{gen_attr}"/>
    <writer name="PythonWriter" class="DatasetWriter"/>
</program>
"""
        xml_path.write_text(cfg, encoding="utf-8")
        rm = colapy.RunManager().load_module("COLA-Py").load_module("COLA_UrQMD")
        rm.load_config(str(xml_path))

        _WORKER_EVENTS.clear()
        rm.run(n_run)

        if len(_WORKER_EVENTS) != n_run:
            err = f"expected {n_run} events from writer, got {len(_WORKER_EVENTS)}"
            for idx in indices:
                results.append({"index": idx, "ok": False, "error": err, "n": 0})
            return results

        for index, ev in zip(indices, _WORKER_EVENTS, strict=True):
            pos, mom, isp = _extract_nucleons(ev.particles)
            n = min(pos.shape[0], mom.shape[0], isp.shape[0])
            pos, mom, isp = pos[:n], mom[:n], isp[:n]
            results.append(
                {
                    "index": index,
                    "ok": True,
                    "n": int(n),
                    "pos": pos,
                    "mom": mom,
                    "is_proton": isp,
                }
            )
    except Exception as e:  # noqa: BLE001
        err = repr(e)
        results = [{"index": index, "ok": False, "error": err, "n": 0} for index in indices]
    finally:
        if not keep_staging:
            shutil.rmtree(staging_dir, ignore_errors=True)

    return results


def _worker_chunk(args: tuple[Any, ...]) -> list[dict[str, Any]]:
    indices, tables_src, phys, keep_staging, staging_parent, chunk_rsd = args
    return _process_chunk(
        list(indices),
        tables_src,
        phys,
        keep_staging,
        staging_parent,
        int(chunk_rsd),
    )


def load_event_npz(path: str | Path) -> tuple[Any, Any, Any]:
    """Load one legacy ``.npz`` shard as ``(pos, mom, is_proton)``."""
    import numpy as np

    z = np.load(path, allow_pickle=False)
    return z["pos"], z["mom"], z["is_proton"]


def load_dataset_pickle(path: str | Path) -> dict[str, Any]:
    """Load ``dataset.pkl`` written by this script (``version``, ``meta``, ``events``, ``failed``)."""
    with Path(path).open("rb") as f:
        return pickle.load(f)


def main() -> int:
    p = argparse.ArgumentParser(description="Generate UrQMD nucleon dataset (parallel).")
    p.add_argument("--n", type=int, default=10_000, help="Number of events (default 10000).")
    p.add_argument(
        "--workers",
        type=int,
        default=6,
        help="Parallel worker processes (default 6); each reuses one RunManager for its chunk.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("datasets/urqmd_nucleons_10k"),
        help="Output directory (created). The bundle is written as <out>/dataset.pkl.",
    )
    p.add_argument(
        "--dataset-file",
        type=Path,
        default=None,
        help="Override output path (single .pkl). Default: <out>/dataset.pkl.",
    )
    p.add_argument(
        "--tables",
        type=Path,
        default=None,
        help="Path to master tables.dat (copied per run). If omitted, search repo.",
    )
    p.add_argument(
        "--staging-dir",
        type=Path,
        default=None,
        help="Parent for per-chunk temp dirs (default: system temp). UrQMD ftn09 is limited to ~77 chars.",
    )
    p.add_argument(
        "--keep-staging",
        action="store_true",
        help="Keep per-chunk temp staging directories (default: remove after each chunk).",
    )
    p.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm. Without tqdm, only a short note is printed (pip install tqdm).",
    )
    p.add_argument("--pro", default="197 79", help="Projectile (A Z).")
    p.add_argument("--tar", default="197 79", help="Target (A Z).")
    p.add_argument("--imp", default="5.", help="Impact parameter.")
    p.add_argument("--elb", default="100.", help="Lab energy / beam parameter (UrQMD XML).")
    p.add_argument("--tim", default="200 200", help="Time window (UrQMD XML).")
    p.add_argument(
        "--nev",
        default="1",
        help="Ignored for dataset generation: chunk size is used as UrQMD ``nev`` and ``run(n)``.",
    )
    p.add_argument(
        "--urqmd-random-base",
        type=int,
        default=None,
        help=(
            "Base for UrQMD ``rsd`` per worker chunk (chunk i uses a deterministic offset). "
            "If omitted, a random base is drawn once so parallel chunks never share the "
            "same clock-based default seed."
        ),
    )
    args = p.parse_args()

    if importlib.util.find_spec("colapy") is None:
        print("ERROR: colapy is not installed or not on PYTHONPATH.", file=sys.stderr)
        return 1
    importlib.import_module("colapy")

    try:
        tables_src = _resolve_tables_src(args.tables)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    staging_parent: str | None = None
    if args.staging_dir is not None:
        sp = args.staging_dir.expanduser().resolve()
        if not sp.is_dir():
            print(f"ERROR: --staging-dir is not a directory: {sp}", file=sys.stderr)
            return 1
        staging_parent = str(sp)

    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = (
        args.dataset_file.expanduser().resolve()
        if args.dataset_file is not None
        else (out_dir / "dataset.pkl")
    )
    if dataset_path.suffix.lower() != ".pkl":
        print("ERROR: --dataset-file must end with .pkl", file=sys.stderr)
        return 1
    dataset_path.parent.mkdir(parents=True, exist_ok=True)

    phys = {
        "pro": args.pro,
        "tar": args.tar,
        "imp": args.imp,
        "elb": args.elb,
        "tim": args.tim,
        "nev": args.nev,
    }

    random_base = args.urqmd_random_base
    if random_base is None:
        random_base = secrets.randbelow(_URQMD_RSD_MOD - 1) + 1

    n_workers = max(1, min(args.workers, args.n))
    chunks = _split_indices(args.n, n_workers)
    chunk_args = [
        (
            ch,
            str(tables_src),
            phys,
            args.keep_staging,
            staging_parent,
            _urqmd_chunk_rsd(random_base, ch[0]),
        )
        for ch in chunks
    ]

    events_slot: list[dict[str, Any] | None] = [None] * args.n
    failed: list[dict[str, Any]] = []

    tqdm_type: Any
    if importlib.util.find_spec("tqdm") is not None:
        tqdm_type = importlib.import_module("tqdm").tqdm
    else:
        tqdm_type = None

    if tqdm_type is None and not args.no_progress:
        print(
            "Note: install tqdm for a progress bar: pip install tqdm",
            file=sys.stderr,
            flush=True,
        )

    # Bar runs only in this process; workers never touch tqdm. It advances when each
    # worker chunk finishes, so it can stay at 0% until the first UrQMD chunk returns.
    use_pbar = tqdm_type is not None and not args.no_progress
    pbar = None
    if use_pbar:
        pbar = tqdm_type(
            total=args.n,
            desc="UrQMD events",
            unit="evt",
            mininterval=0.3,
            file=sys.stderr,
            disable=False,
            dynamic_ncols=True,
        )
        pbar.refresh()

    cum_ok = 0
    cum_fail = 0

    with ProcessPoolExecutor(max_workers=len(chunk_args)) as ex:
        futures = [ex.submit(_worker_chunk, ca) for ca in chunk_args]
        future_n = {f: len(ca[0]) for f, ca in zip(futures, chunk_args)}
        for fut in as_completed(futures):
            n_chunk = future_n[fut]
            try:
                batch = fut.result()
            except Exception as e:  # noqa: BLE001
                failed.append({"error": repr(e)})
                cum_fail += n_chunk
                if pbar is not None:
                    pbar.update(n_chunk)
                    pbar.set_postfix(ok=cum_ok, failed=cum_fail, refresh=True)
                continue
            for r in batch:
                if r.get("ok"):
                    idx = int(r["index"])
                    events_slot[idx] = {
                        "pos": r["pos"],
                        "mom": r["mom"],
                        "is_proton": r["is_proton"],
                    }
                else:
                    failed.append(r)
            ok_batch = sum(1 for r in batch if r.get("ok"))
            cum_ok += ok_batch
            cum_fail += len(batch) - ok_batch
            if pbar is not None:
                pbar.update(n_chunk)
                pbar.set_postfix(ok=cum_ok, failed=cum_fail, refresh=True)

    if pbar is not None:
        pbar.close()

    if use_pbar:
        tqdm_type.write(f"Writing {dataset_path} ...")
    else:
        print(f"Writing {dataset_path} ...", flush=True)

    n_written = sum(1 for e in events_slot if e is not None)
    meta = {
        "format": "urqmd_nucleon_dataset_v1",
        "n_requested": args.n,
        "n_written": n_written,
        "n_failed": len(failed),
        "worker_processes": len(chunk_args),
        "chunks": [len(c) for c in chunks],
        "executor": "processes",
        "physics": phys,
        "out_dir": str(out_dir),
        "dataset_path": str(dataset_path),
        "staging_parent": staging_parent or tempfile.gettempdir(),
        "tables_source": str(tables_src),
        "keep_staging": args.keep_staging,
        "urqmd_random_base": random_base,
        "urqmd_rsd_per_chunk": [_urqmd_chunk_rsd(random_base, ch[0]) for ch in chunks],
    }
    bundle: dict[str, Any] = {
        "version": 1,
        "meta": meta,
        "events": events_slot,
        "failed": failed[:500],
    }
    with dataset_path.open("wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
    nf = len(failed)
    if nf:
        print(
            f"Done: {n_written} / {args.n} events in {dataset_path}\n"
            f"Events with errors: {nf} (details in pickle key 'failed', up to 500 entries).",
            flush=True,
        )
    else:
        print(f"Done: {n_written} / {args.n} events in {dataset_path}", flush=True)
    return 0 if n_written == args.n else 2


if __name__ == "__main__":
    raise SystemExit(main())
