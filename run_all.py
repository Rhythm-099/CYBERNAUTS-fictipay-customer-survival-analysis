import subprocess
import sys

scripts = [
    "01_build_features.py",
    "02_train_models.py",
    "03_optimize_and_submit.py",
    "04_validate_submission.py"
]

for script in scripts:
    print(f"Running {script}")
    result = subprocess.run([sys.executable, script])
    if result.returncode != 0:
        raise SystemExit(result.returncode)
