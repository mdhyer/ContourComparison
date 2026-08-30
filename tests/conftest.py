import pytest
import tempfile
import numpy as np
import soundfile as sf
from pathlib import Path
import scipy.io

try:
    import matlab.engine
    HAS_MATLAB = True
except ImportError:
    HAS_MATLAB = False


def pytest_configure(config):
    config.addinivalue_line("markers", "matlab: tests requiring a MATLAB installation")


def pytest_collection_modifyitems(config, items):
    for item in items:
        if "matlab" in item.keywords and not HAS_MATLAB:
            item.add_marker(pytest.mark.skip(reason="MATLAB not available"))


def pytest_addoption(parser):
    parser.addoption(
        "--config-path", action="store", default=None,
        help="Path to an existing PipelineConfig JSON file to use for parity tests"
    )

@pytest.fixture
def config():
    """Provide a default PipelineConfig for tests."""
    from audio_analysis.config import PipelineConfig
    return PipelineConfig()

@pytest.fixture
def temp_audio_dir():
    """Create a temporary directory for audio I/O tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

@pytest.fixture
def synthetic_wav(temp_audio_dir):
    """Generate a synthetic mono WAV file for testing."""
    sr = 96000
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # 1 kHz sine wave at 0.5 amplitude
    freq = 1000
    signal = 0.5 * np.sin(2 * np.pi * freq * t)
    
    wav_path = temp_audio_dir / "test_signal.wav"
    sf.write(str(wav_path), signal, sr)
    return wav_path, sr, duration

@pytest.fixture
def multi_fbid_data_dir(temp_audio_dir):
    """
    Create a temporary directory structure matching the canonical multi-FBID, multi-SNR layout.
    Structure:
    temp_audio_dir/
    ├── data/
    │   └── NoiseLevels/
    │       ├── CLEAN/
    │       │   ├── FBID_001/
    │       │   └── FBID_002/
    │       └── SNR_20/
    │           ├── FBID_001/
    │           └── FBID_002/
    ├── Params/
    │   ├── FBID_001/
    │   └── FBID_002/
    └── PrecomputeContours/
        ├── CREPE/
        │   ├── CLEAN/
        │   │   ├── FBID_001/
        │   │   └── FBID_002/
        │   └── SNR_20/
        └── Silbido Profundo/
    """
    fbids = ["FBID_001", "FBID_002"]
    snr_levels = ["CLEAN", "SNR_20"]

    # 1. Create data/NoiseLevels/ structure per README
    data_root = temp_audio_dir / "data" / "NoiseLevels"
    for snr in snr_levels:
        snr_dir = data_root / snr
        snr_dir.mkdir(parents=True, exist_ok=True)
        for fbid in fbids:
            fbid_dir = snr_dir / fbid
            fbid_dir.mkdir(parents=True, exist_ok=True)
            for i in range(2):
                wav_path = fbid_dir / f"whistle_{i}.wav"
                sr = 48000
                duration = 1.0
                t = np.linspace(0, duration, int(sr * duration), endpoint=False)
                signal = 0.5 * np.sin(2 * np.pi * 1000 * t)
                sf.write(str(wav_path), signal, sr)

    # 2. Create Params/ structure (unified ground truth & params)
    params_dir = temp_audio_dir / "Params"
    for fbid in fbids:
        fbid_params = params_dir / fbid
        fbid_params.mkdir(parents=True, exist_ok=True)
        for i in range(2):
            stem = f"whistle_{i}"
            dummy_mat = fbid_params / f"{stem}_params.mat"
            scipy.io.savemat(str(dummy_mat), {'W': {'contour': np.array([[0, 1000], [1, 1000]]), 'discont': []}})

    # 3. Create PrecomputeContours/ structure
    pc_dir = temp_audio_dir / "PrecomputeContours"
    for algo in ["CREPE", "Silbido Profundo"]:
        algo_dir = pc_dir / algo
        for snr in snr_levels:
            snr_pc_dir = algo_dir / snr
            snr_pc_dir.mkdir(parents=True, exist_ok=True)
            for fbid in fbids:
                (snr_pc_dir / fbid).mkdir(parents=True, exist_ok=True)

    return temp_audio_dir
