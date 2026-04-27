from pathlib import Path
import runpy


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "generate_spelling_data.py"


if __name__ == "__main__":
    runpy.run_path(SCRIPT_PATH, run_name="__main__")
