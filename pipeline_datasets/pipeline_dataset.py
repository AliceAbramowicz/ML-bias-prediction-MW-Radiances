import subprocess
import yaml
import argparse
import json
import os
from pathlib import Path

# Run this pipeline using the command line: python3 pipeline_dataset.py --config config/CONFIG_FILE.yaml

def run_script(script, env):
    """Run a Python or Bash script with the given environment."""
    print(f"Running {script} ...")
    if script.endswith(".py"):
        if script.endswith("varbc_dataset_preparation.py"):
            cmd = ["python3", script, "-i", env["DEST_VARBC_DIR"], "-o", env["VARBC_OUTPUT_FILE"]]
        else:
            cmd = ["python3", script]
    elif script.endswith(".sh"):
        cmd = ["bash", script]
    else:
        raise ValueError(f"Unknown script type: {script}")
    
    subprocess.run(cmd, check=True, env=env)

def main(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    env = os.environ.copy()

    env.update({
        "DOMAIN": cfg["domain"],
        "EXP": cfg["EXP"],
        
        "HM_DIR": cfg["hm_dir"],
        "DATA_DIR": cfg["data_dir"],
        "RAW_DATA_DIR": cfg["raw_data_dir"],
        "CCMA_OUTPUT_DIR": cfg["ccma_output_dir"],
        "SAT_TABLES_DIR": cfg["sat_tables_dir"],
        "ANCHORS_DIR": cfg["anchors_dir"],
        "MERGED_ODB_DIR": cfg["merged_odb_dir"],

        "ORIGIN_VARBC_DIR": cfg["origin_varbc_dir"],
        "DEST_VARBC_DIR": cfg["dest_varbc_dir"],
        "VARBC_OUTPUT_FILE": cfg["varbc_output_file"],

        "OUTPUT_DIR": cfg["output_dir"],
        "FINAL_DF": ["final_df"],

        "YEAR": str(cfg["year"]),
        "MONTHS": " ".join(str(m) for m in cfg.get("months", [])),
        "CYCLES": " ".join(str(c) for c in cfg.get("cycles", [])),
        "DAYS_JSON": json.dumps(cfg.get("days", {}))
    })

    for k, v in env.items():
        if isinstance(v, list):
            env[k] = " ".join(str(i) for i in v)
        elif not isinstance(v, str):
            env[k] = str(v)


    # Ensure all directories exist
    for key in ["DATA_DIR", "CCMA_OUTPUT_DIR", "SAT_TABLES_DIR", "ANCHORS_DIR", "MERGED_ODB_DIR", "DEST_VARBC_DIR", "OUTPUT_DIR"]:
        Path(env[key]).mkdir(parents=True, exist_ok=True)


    for script in cfg['scripts']:
        run_script(script, env)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to dataset config YAML")
    args = parser.parse_args()

    main(args.config)

