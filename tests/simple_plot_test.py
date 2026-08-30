import sys
import matplotlib.pyplot as plt
from pathlib import Path

# Ensure the project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.audio_analysis.plotting.plotting import visualize_together
from src.audio_analysis.config import config

def run_visualization_test():
    # ─────────────────────────────────────────────────────────────
    # CONFIGURATION SETUP
    # Update these paths to point to your actual audio & ground truth data
    # ─────────────────────────────────────────────────────────────
    CONFIG = {
        "wav_file": str(PROJECT_ROOT / "project" / "data" / "F151" / "F151-2006-SW-IND50658.wav"),
        "params_dir": str(PROJECT_ROOT / "project" / "Params"),
        "model_dir": str(PROJECT_ROOT / "project" / "models"),
        "wav_src": None,               # Optional: parent dir containing wav files
        "load_src": None,              # Optional: dir with precomputed contour files
        "use_precomputed": False,      # Set True to skip algorithm execution & load from disk
        "algorithms": ["CREPE"]
    }

    # List of CREPE model weights to test sequentially
    CREPE_MODELS = [
        "model_only-0_dolphins.pth",
        "model_only-0_bottlenose_dolphins.pth"
    ]

    # Apply model_dir to global config so CrepePredictor can locate weights
    config.paths.model_dir = Path(CONFIG["model_dir"])

    # Basic path validation
    if not Path(CONFIG["wav_file"]).exists():
        raise FileNotFoundError(f"Audio file not found: {CONFIG['wav_file']}")
    if not Path(CONFIG["params_dir"]).exists():
        raise FileNotFoundError(f"Params directory not found: {CONFIG['params_dir']}")
    if not Path(CONFIG["model_dir"]).exists():
        raise FileNotFoundError(f"Model directory not found: {CONFIG['model_dir']}")

    for model_name in CREPE_MODELS:
        print(f"\n▶ Running visualization for CREPE model: {model_name}")
        print(f"  Audio: {CONFIG['wav_file']}")
        print(f"  Params: {CONFIG['params_dir']}")
        print(f"  Models: {CONFIG['model_dir']}")
        
        # Update global config for the current model weights
        config.algorithm.crepe_model_name = model_name

        try:
            visualize_together(
                wav=CONFIG["wav_file"],
                params_dir=CONFIG["params_dir"],
                algorithms=CONFIG["algorithms"],
                wav_src=CONFIG["wav_src"],
                use_precomputed=CONFIG["use_precomputed"],
                load_src=CONFIG["load_src"]
            )
            print(f"✅ Visualization completed successfully for {model_name}.")
        except Exception as e:
            print(f"❌ Error during visualization for {model_name}: {e}")
            continue

    # Display all generated figures
    plt.show()
    print("\n✅ All visualizations completed.")

if __name__ == "__main__":
    run_visualization_test()
