#!/usr/bin/env python3

"""
Pipeline

    python3 pipeline_dataset.py --config config/dini.yaml
    python3 pipeline_dataset.py --config config/dini.yaml --dry-run

Tasks:
- derive every dataset path from a single root_dir
- expand `start_date`/`end_date`/`cycles` into list of YYYYMMDDHH stamps
- validate required keys
- hand all of it to the scripts through the environment

Scripts receive the environment variables
"""

import argparse
import datetime as dt
import os
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent

REQUIRED_KEYS = ("domain", "EXP", "cycles", "scripts", "sensors", "satellites")
MANAGED_DIRS = ("DATA_DIR", "CCMA_OUTPUT_DIR", "SAT_TABLES_DIR", "OUTPUT_DIR")

def _as_date(value, label):
    """Accept a date, datetime or ISO string."""
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    try:
        return dt.date.fromisoformat(str(value).strip())
    except ValueError as exc:
        raise ValueError(
            f"{label} must be a date (YYYY-MM-DD), got {value!r}") from exc

def load_config(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"{config_path} did not parse to a mapping")
    missing = [k for k in REQUIRED_KEYS if not cfg.get(k)]
    if missing:
        raise KeyError(f"{config_path} is missing required key(s): {', '.join(missing)}")
    return cfg

def collect_ranges(cfg):
    """Return [(start, end), ...] from date_ranges, or start_date/end_date."""
    if cfg.get("date_ranges"):
        ranges = []
        for i, item in enumerate(cfg["date_ranges"]):
            if not isinstance(item, dict) or "start" not in item or "end" not in item:
                raise ValueError(
                    f"date_ranges[{i}] must be a mapping with 'start' and 'end'")
            start = _as_date(item["start"], f"date_ranges[{i}].start")
            end = _as_date(item["end"], f"date_ranges[{i}].end")
            if end < start:
                raise ValueError(
                    f"date_ranges[{i}]: end ({end}) is before start ({start})")
            ranges.append((start, end))
        return ranges

    if cfg.get("start_date") and cfg.get("end_date"):
        start = _as_date(cfg["start_date"], "start_date")
        end = _as_date(cfg["end_date"], "end_date")
        if end < start:
            raise ValueError(f"end_date ({end}) is before start_date ({start})")
        return [(start, end)]
    
    raise KeyError("config needs either 'date_ranges' or 'start_date' and 'end_date'")

def expand_datetimes(cfg):
    """All configured ranges"""
    ranges = collect_ranges(cfg)
    excluded = {_as_date(d, "exclude_dates entry")
                for d in (cfg.get("exclude_dates") or [])}
    cycles = [str(c).zfill(2) for c in cfg["cycles"]]
    if not cycles:
        raise ValueError("cycles empty")

    days = set()
    for start, end in ranges:
        day = start
        while day <= end:
            if day not in excluded:
                days.add(day)
            day += dt.timedelta(days=1)

    stamps = [f"{day:%Y%m%d}{cy}" for day in sorted(days) for cy in cycles]
    return ranges, cycles, sorted(days), stamps

def period_label(days):
    years = sorted({day.year for day in days})
    return str(years[0]) if len(years) == 1 else f"{years[0]}-{years[-1]}"
 
 
def resolve_paths(cfg):
    """Derive every path from root_dir. Any key from the YAML wins."""
    root = str(cfg.get("root_dir") or REPO_ROOT)
    domain = cfg["domain"]
    exp = str(cfg["EXP"])
    data = str(Path(root).resolve().parent / "datasets")
    outdir = str(f"{data}/final_df")
 
    derived = {
        "root_dir":          root,
        "data_dir":          data,
        "output_dir":        outdir,
        "ccma_output_dir":   f"{data}/ccma_{exp.lower()}",
        "sat_tables_dir":    f"{data}/sat_{exp.lower()}_tables",
        "varbc_output_file": f"{data}/varbc_{exp.lower()}_common_sat_sen.csv",
        "final_df":          f"{outdir}/big_df_stats_{exp.lower()}.csv",
    }
    for key, value in derived.items():
        if not cfg.get(key):
            cfg[key] = value
    return cfg
 
 
def build_env(cfg, cycles, stamps, period):
    env = os.environ.copy()
    env.update({
        "DOMAIN":            str(cfg["domain"]),
        "EXP":               str(cfg["EXP"]),
 
        "ROOT_DIR":          str(cfg["root_dir"]),
        "DATA_DIR":          str(cfg["data_dir"]),
        "RAW_DATA_DIR":      str(cfg["raw_data_odb"]),
        "RAW_DATA_ODB":      str(cfg["raw_data_odb"]),
        "RAW_DATA_VARBC":    str(cfg["raw_data_varbc"]),
        "VARBC_PATTERN":     str(cfg["varbc_pattern"]),
        "CCMA_OUTPUT_DIR":   str(cfg["ccma_output_dir"]),
        "SAT_TABLES_DIR":    str(cfg["sat_tables_dir"]),
 
        "VARBC_OUTPUT_FILE": str(cfg["varbc_output_file"]),
 
        "OUTPUT_DIR":        str(cfg["output_dir"]),
        "FINAL_DF":          str(cfg["final_df"]),
        "PERIOD":            period,
        "CYCLES":            " ".join(cycles),
        "DATETIMES":         " ".join(stamps),
 
        "SENSORS":           str(cfg["sensors"]),
        "SATELLITES":        str(cfg["satellites"]),
    })
    return env
 
 
def run_script(script, env):
    """Run a python/bash script with the given environment."""
    script_path = (REPO_ROOT / script).resolve()
    if not script_path.is_file():
        raise FileNotFoundError(f"Script not found: {script_path}")
 
    if script_path.suffix == ".py":
        cmd = [sys.executable, str(script_path)]
    elif script_path.suffix == ".sh":
        cmd = ["bash", str(script_path)]
    else:
        raise ValueError(f"Unknown script type: {script}")
 
    print(f"######## Running {script} ########", flush=True)
    subprocess.run(cmd, check=True, env=env, cwd=str(REPO_ROOT))
 
 
def main(config_path, dry_run=False):
    cfg = load_config(config_path)
 
    ranges, cycles, days, stamps = expand_datetimes(cfg)
    period = period_label(days)
    cfg = resolve_paths(cfg)
    env = build_env(cfg, cycles, stamps, period)
 
    print(f"Domain {env['DOMAIN']}, period {period}, {len(stamps)} timestamps ({len(days)} days x {len(cycles)} cycles)")
    for start, end in ranges:
        print(f"range {start} ... {end}")
 
    if dry_run:
        for key in ("ROOT_DIR", "PERIOD", "DATA_DIR", "RAW_DATA_ODB",
                    "RAW_DATA_VARBC", "VARBC_PATTERN",
                    "CCMA_OUTPUT_DIR", "SAT_TABLES_DIR", "VARBC_OUTPUT_FILE",
                    "OUTPUT_DIR", "FINAL_DF", "SENSORS", "SATELLITES"):
            print(f"{key:<18} {env[key]}")
        print(f"{'DATETIMES':<18} {stamps[0]} ... {stamps[-1]}")
        print("Scripts:")
        for script in cfg["scripts"]:
            path = (REPO_ROOT / script).resolve()
            print(f"[{'ok ' if path.is_file() else 'MISSING'}] {script}")
        return

    for key in MANAGED_DIRS:
        Path(env[key]).mkdir(parents=True, exist_ok=True)
    Path(env["VARBC_OUTPUT_FILE"]).parent.mkdir(parents=True, exist_ok=True)

    for script in cfg["scripts"]:
        run_script(script, env)
    
    print("Pipeline finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the dataset pipeline")
    parser.add_argument("--config", required=True,
                        help="Path to dataset config YAML")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print resolved config and exit without running")
    args = parser.parse_args()
 
    main(args.config, dry_run=args.dry_run)
 