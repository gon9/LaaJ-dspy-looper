"""Observabilityのテスト"""

import json
import tempfile

from laaj.observability import LocalLogger


def test_log_experiment():
    """実験ログの記録テスト"""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = LocalLogger(output_dir=tmpdir)

        config = {"optimizer": "MIPROv2", "model": "gpt-4o-mini"}
        results = {"test_score": 0.85}

        log_file = logger.log_experiment(experiment_name="test_exp", config=config, results=results)

        assert log_file.exists()

        with open(log_file, encoding="utf-8") as f:
            log_data = json.load(f)

        assert log_data["experiment_name"] == "test_exp"
        assert log_data["config"] == config
        assert log_data["results"] == results


def test_log_evaluation():
    """評価ログの記録テスト"""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = LocalLogger(output_dir=tmpdir)

        scores = {"test_score": 0.90}

        log_file = logger.log_evaluation(
            experiment_name="test_exp", module_type="baseline", scores=scores
        )

        assert log_file.exists()

        with open(log_file, encoding="utf-8") as f:
            log_data = json.load(f)

        assert log_data["experiment_name"] == "test_exp"
        assert log_data["module_type"] == "baseline"
        assert log_data["scores"] == scores


def test_get_experiment_logs():
    """実験ログの取得テスト"""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = LocalLogger(output_dir=tmpdir)

        logger.log_experiment("exp1", {}, {})
        logger.log_experiment("exp2", {}, {})
        logger.log_experiment("exp1", {}, {})

        all_logs = logger.get_experiment_logs()
        assert len(all_logs) == 3

        exp1_logs = logger.get_experiment_logs(experiment_name="exp1")
        assert len(exp1_logs) == 2
        assert all(log["experiment_name"] == "exp1" for log in exp1_logs)
