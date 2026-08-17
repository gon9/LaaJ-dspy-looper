"""Observabilityのテスト"""

import tempfile

from laaj.config import ObservabilityConfig
from laaj.observability import LocalLogger, ObservabilityFactory


def test_log_experiment():
    """実験結果のログ記録をテスト"""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = LocalLogger(output_dir=tmpdir)

        config = {
            "experiment": {"name": "test_exp"},
            "lm": {"model": "gpt-4o-mini"},
        }
        results = {"test_score": 0.85}

        log_file = logger.log_experiment(
            experiment_name="test_exp",
            config=config,
            results=results,
            optimized_module_path="/path/to/module",
        )

        assert log_file.exists()
        logs = logger.get_experiment_logs("test_exp")
        assert len(logs) == 1
        assert logs[0]["results"]["test_score"] == 0.85


def test_log_evaluation():
    """評価結果のログ記録をテスト"""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = LocalLogger(output_dir=tmpdir)

        log_file = logger.log_evaluation(
            experiment_name="test_exp",
            module_type="baseline",
            scores={"test_score": 0.75},
        )

        assert log_file.exists()


def test_get_experiment_logs():
    """実験ログの取得をテスト"""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = LocalLogger(output_dir=tmpdir)

        logger.log_experiment(
            experiment_name="exp1",
            config={},
            results={"score": 0.8},
        )
        logger.log_experiment(
            experiment_name="exp2",
            config={},
            results={"score": 0.9},
        )

        logs_all = logger.get_experiment_logs()
        assert len(logs_all) == 2

        logs_exp1 = logger.get_experiment_logs("exp1")
        assert len(logs_exp1) == 1
        assert logs_exp1[0]["experiment_name"] == "exp1"


def test_observability_factory():
    """ObservabilityFactoryの生成切り替えをテスト"""
    # Local
    local_cfg = ObservabilityConfig(backend="local")
    backend = ObservabilityFactory.create_backend(local_cfg)
    assert isinstance(backend, LocalLogger)
    assert backend.initialize() is True

    # Langfuse (未設定時はLocalLoggerにフォールバック)
    lf_unconfigured = ObservabilityConfig(backend="langfuse", langfuse_public_key=None)
    backend_fb = ObservabilityFactory.create_backend(lf_unconfigured)
    assert isinstance(backend_fb, LocalLogger)
