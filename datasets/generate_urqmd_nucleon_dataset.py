#!/usr/bin/env python3
import argparse
import contextlib
import importlib.util
import os
import pickle
import secrets
import shutil
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from datasets.urqmd_writer import WORKER_EVENTS as _WORKER_EVENTS
from datasets.urqmd_writer import WRITER_CLASS as _WRITER_CLASS

_CLUSTERING_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _CLUSTERING_DIR.parent

_URQMD_RSD_MOD = (1 << 31) - 2

PAPER_PRESETS: dict[str, dict[str, str]] = {
    "au_ag_10p6gev": {
        "label": "197Au + 107Ag @ 10.6A GeV (Sec. 3 validation)",
        "pro": "197 79",
        "tar": "107 47",
        "elb": "10.6",
        "tim": "100 100",
        "imp": "5.",
        "out_subdir": "svetlichnyi_au_ag_10p6gev",
    },
    "sn_sn_600mev": {
        "label": "124Sn + 124Sn @ 600A MeV (Sec. 3 validation)",
        "pro": "124 50",
        "tar": "124 50",
        "elb": "0.6",
        "tim": "100 100",
        "imp": "5.",
        "out_subdir": "svetlichnyi_sn_sn_600mev",
    },
    "xe_xe_3p8gev": {
        "label": "124Xe + 130Xe @ 3.8A GeV (Sec. 4 NICA prediction)",
        "pro": "124 54",
        "tar": "130 54",
        "elb": "3.8",
        "tim": "100 100",
        "imp": "5.",
        "out_subdir": "svetlichnyi_xe_xe_3p8gev",
    },
    "xe_w_3p8gev": {
        "label": "124Xe + 184W @ 3.8A GeV (Sec. 4 NICA prediction)",
        "pro": "124 54",
        "tar": "184 74",
        "elb": "3.8",
        "tim": "100 100",
        "imp": "5.",
        "out_subdir": "svetlichnyi_xe_w_3p8gev",
    },
}

URQMD_MODES: dict[str, dict[str, str]] = {
    "cascade": {"eos": "0", "label": "UrQMD cascade (eos=0)"},
    "skyrme": {"eos": "1", "label": "UrQMD Skyrme (eos=1)"},
}

@contextlib.contextmanager
def _quiet_urqmd_stdout() -> Any:
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield

def _urqmd_chunk_rsd(random_base: int, global_first_index: int) -> int:
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
        _REPO_ROOT / "tables.dat",
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

def _process_chunk(
    indices: list[int],
    tables_src: str,
    phys: dict[str, str],
    keep_staging: bool,
    staging_parent: str | None,
    chunk_rsd: int,
) -> list[dict[str, Any]]:
    _ensure_importable_from_repo_root()
    import colapy

    pro, tar, imp, elb, tim, eos = (
        phys["pro"],
        phys["tar"],
        phys["imp"],
        phys["elb"],
        phys["tim"],
        phys.get("eos", "0"),
    )
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
        eos="{eos}"
        rsd="{chunk_rsd}"
        generated_config_file="{gen_attr}"/>
    <writer name="PythonWriter" class="{_WRITER_CLASS}"/>
</program>
"""
        xml_path.write_text(cfg, encoding="utf-8")
        rm = colapy.RunManager().load_module("COLA-Py").load_module("COLA_UrQMD")
        rm.load_config(file=str(xml_path))

        _WORKER_EVENTS.clear()
        with _quiet_urqmd_stdout():
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
    import numpy as np

    z = np.load(path, allow_pickle=False)
    return z["pos"], z["mom"], z["is_proton"]

def load_dataset_pickle(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as f:
        return pickle.load(f)

def _ensure_importable_from_repo_root() -> None:
    root = str(_REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)

def _print_presets() -> None:
    for key, cfg in PAPER_PRESETS.items():
        print(f"{key}:")
        print(f"  {cfg['label']}")
        print(
            f"  pro={cfg['pro']} tar={cfg['tar']} elb={cfg['elb']} "
            f"tim={cfg['tim']} imp={cfg['imp']}"
        )
        for mode in ("cascade", "skyrme"):
            print(f"  -> datasets/{cfg['out_subdir']}_{mode}/dataset.pkl")

def _modes_for_arg(urqmd_mode: str) -> list[str]:
    if urqmd_mode == "both":
        return ["cascade", "skyrme"]
    return [urqmd_mode]

def _run_one_dataset(
    *,
    n_events: int,
    n_workers: int,
    phys: dict[str, str],
    out_dir: Path,
    dataset_path: Path,
    tables_src: Path,
    staging_parent: str | None,
    keep_staging: bool,
    random_base: int,
    no_progress: bool,
    progress_label: str,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_path.parent.mkdir(parents=True, exist_ok=True)

    n_workers = max(1, min(n_workers, n_events))
    chunks = _split_indices(n_events, n_workers)
    chunk_sizes = [len(c) for c in chunks]
    if not no_progress:
        print(
            f"  {progress_label}: {n_events} events in {len(chunks)} process chunks "
            f"(sizes {chunk_sizes}), 1 UrQMD RunManager per chunk",
            flush=True,
        )
    chunk_args = [
        (
            ch,
            str(tables_src),
            phys,
            keep_staging,
            staging_parent,
            _urqmd_chunk_rsd(random_base, ch[0]),
        )
        for ch in chunks
    ]

    events_slot: list[dict[str, Any] | None] = [None] * n_events
    failed: list[dict[str, Any]] = []

    tqdm_type: Any
    if importlib.util.find_spec("tqdm") is not None:
        tqdm_type = importlib.import_module("tqdm").tqdm
    else:
        tqdm_type = None

    if tqdm_type is None and not no_progress:
        print(
            "Note: install tqdm for a progress bar: pip install tqdm",
            file=sys.stderr,
            flush=True,
        )

    use_pbar = tqdm_type is not None and not no_progress
    pbar = None
    if use_pbar:
        pbar = tqdm_type(
            total=n_events,
            desc=progress_label,
            unit="evt",
            mininterval=0.3,
            file=sys.stderr,
            disable=False,
            dynamic_ncols=True,
        )
        pbar.refresh()

    cum_ok = 0
    cum_fail = 0
    t0 = time.perf_counter()

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
                elapsed = time.perf_counter() - t0
                rate = cum_ok / elapsed if elapsed > 0 else 0.0
                pbar.set_postfix(
                    ok=cum_ok,
                    failed=cum_fail,
                    evt_s=f"{rate:.2f}" if rate > 0 else "?",
                    refresh=True,
                )

    if pbar is not None:
        pbar.close()
    elapsed_total = time.perf_counter() - t0
    if not no_progress and cum_ok > 0:
        print(
            f"  {progress_label}: generated {cum_ok} events in {elapsed_total:.1f}s "
            f"({cum_ok / elapsed_total:.2f} evt/s wall)",
            flush=True,
        )

    if use_pbar:
        tqdm_type.write(f"Writing {dataset_path} ...")
    else:
        print(f"Writing {dataset_path} ...", flush=True)

    n_written = sum(1 for e in events_slot if e is not None)
    meta = {
        "format": "urqmd_nucleon_dataset_v1",
        "n_requested": n_events,
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
        "keep_staging": keep_staging,
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
            f"Done: {n_written} / {n_events} events in {dataset_path}\n"
            f"Events with errors: {nf} (details in pickle key 'failed', up to 500 entries).",
            flush=True,
        )
    else:
        print(f"Done: {n_written} / {n_events} events in {dataset_path}", flush=True)
    return 0 if n_written == n_events else 2

def main() -> int:
    _ensure_importable_from_repo_root()

    p = argparse.ArgumentParser(description="Generate UrQMD nucleon dataset (parallel).")
    p.add_argument(
        "--preset",
        choices=sorted(PAPER_PRESETS),
        default=None,
        help="Collision system from Svetlichnyi UrQMD-AMC paper (overrides pro/tar/elb/tim/imp).",
    )
    p.add_argument(
        "--list-presets",
        action="store_true",
        help="Print paper collision presets and exit.",
    )
    p.add_argument("--n", type=int, default=10_000, help="Number of events (default 10000).")
    p.add_argument(
        "--workers",
        type=int,
        default=None,
        help=(
            "Parallel worker processes (default: os.cpu_count()); each process runs one "
            "UrQMD RunManager for its chunk (spawn, not threads)."
        ),
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory (created). Default: datasets/<preset>/ or datasets/urqmd_nucleons_10k.",
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
        "--urqmd-mode",
        choices=("cascade", "skyrme", "both"),
        default="cascade",
        help=(
            "UrQMD equation-of-state flag passed as ``eos`` in the generated input: "
            "cascade (0) or Skyrme (1). ``both`` writes separate datasets (paper default)."
        ),
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

    if args.list_presets:
        _print_presets()
        return 0

    if args.dataset_file is not None and args.urqmd_mode == "both":
        print("ERROR: --dataset-file cannot be used with --urqmd-mode both", file=sys.stderr)
        return 1

    preset_cfg: dict[str, str] | None = None
    if args.preset is not None:
        preset_cfg = PAPER_PRESETS[args.preset]
        args.pro = preset_cfg["pro"]
        args.tar = preset_cfg["tar"]
        args.elb = preset_cfg["elb"]
        args.tim = preset_cfg["tim"]
        args.imp = preset_cfg["imp"]

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

    random_base = args.urqmd_random_base
    if random_base is None:
        random_base = secrets.randbelow(_URQMD_RSD_MOD - 1) + 1

    n_workers = args.workers if args.workers is not None else max(1, os.cpu_count() or 6)

    modes = _modes_for_arg(args.urqmd_mode)
    exit_code = 0

    for mode_i, mode in enumerate(modes):
        mode_info = URQMD_MODES[mode]
        if args.out is not None:
            out_dir = args.out.resolve()
            if len(modes) > 1:
                out_dir = out_dir.parent / f"{out_dir.name}_{mode}"
        elif preset_cfg is not None:
            out_dir = _REPO_ROOT / "datasets" / f"{preset_cfg['out_subdir']}_{mode}"
        else:
            out_dir = _REPO_ROOT / "datasets" / f"urqmd_nucleons_{args.n}_{mode}"

        dataset_path = (
            args.dataset_file.expanduser().resolve()
            if args.dataset_file is not None
            else (out_dir / "dataset.pkl")
        )
        if dataset_path.suffix.lower() != ".pkl":
            print("ERROR: --dataset-file must end with .pkl", file=sys.stderr)
            return 1

        phys = {
            "pro": args.pro,
            "tar": args.tar,
            "imp": args.imp,
            "elb": args.elb,
            "tim": args.tim,
            "nev": args.nev,
            "eos": mode_info["eos"],
            "urqmd_mode": mode,
        }
        if preset_cfg is not None:
            phys["paper_preset"] = args.preset
            phys["paper_label"] = preset_cfg["label"]
        phys["urqmd_mode_label"] = mode_info["label"]

        mode_seed = _urqmd_chunk_rsd(random_base, mode_i * 10_007)
        preset_tag = args.preset or "custom"
        progress_label = f"{preset_tag} {mode}"

        print(
            f"\n=== {progress_label} | eos={phys['eos']} | n={args.n} -> {dataset_path} ===",
            flush=True,
        )
        rc = _run_one_dataset(
            n_events=args.n,
            n_workers=n_workers,
            phys=phys,
            out_dir=out_dir,
            dataset_path=dataset_path,
            tables_src=tables_src,
            staging_parent=staging_parent,
            keep_staging=args.keep_staging,
            random_base=mode_seed,
            no_progress=args.no_progress,
            progress_label=progress_label,
        )
        exit_code = max(exit_code, rc)

    return exit_code

if __name__ == "__main__":
    if __package__ is None:
        print(
            "Hint: run from repo root as  python -m datasets.generate_urqmd_nucleon_dataset  ...",
            file=sys.stderr,
        )
    raise SystemExit(main())
