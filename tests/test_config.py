import pytest
from audio_analysis.config import PipelineConfig, MetricsConfig, CoverageMode, FreqDiffMode, FragMode

class TestConfig:
    def test_default_config(self):
        """Verify default configuration values."""
        cfg = PipelineConfig()
        assert cfg.metrics.coverage_mode == CoverageMode.TOTAL
        assert cfg.audio.sample_rate == 48000

    def test_custom_metrics_config(self):
        """Verify custom metrics configuration overrides."""
        cfg = MetricsConfig(coverage_mode=CoverageMode.FFC, freq_diff_threshold=0.3)
        assert cfg.coverage_mode == CoverageMode.FFC
        assert cfg.freq_diff_threshold == 0.3

    def test_invalid_coverage_mode(self):
        """Ensure Pydantic raises ValueError for invalid enum values."""
        with pytest.raises(ValueError):
            MetricsConfig(coverage_mode="invalid_mode")

    def test_freq_diff_mode_enum(self):
        """Verify FreqDiffMode enum parsing."""
        cfg = MetricsConfig(freq_diff_mode=FreqDiffMode.FFC)
        assert cfg.freq_diff_mode == FreqDiffMode.FFC