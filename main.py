"""Optional launcher for ResearchMind AI's Streamlit application."""

import subprocess
import sys
from pathlib import Path


def main() -> None:
    """Launch the ResearchMind AI Streamlit UI from the project directory."""
    project_dir = Path(__file__).resolve().parent
    app_path = project_dir / "app.py"
    if not app_path.is_file():
        raise SystemExit(
            "Cannot find app.py. Place main.py in the ResearchMind AI project root "
            "alongside app.py."
        )

    print("ResearchMind AI — Agentic Research Intelligence Assistant")
    print("Explore Research. Validate Knowledge. Discover Insights.")
    try:
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", str(app_path)],
            cwd=project_dir,
            check=True,
        )
    except KeyboardInterrupt:
        print("\nResearchMind AI stopped.")
    except FileNotFoundError as exc:
        raise SystemExit("Python executable not found.") from exc
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"Streamlit exited with status {exc.returncode}.") from exc


if __name__ == "__main__":
    main()
