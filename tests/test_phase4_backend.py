"""Phase 4 tests: PostgreSQL URL normalization, MLflow tracking, checkpoint resume, and job APIs."""
from __future__ import annotations

import os
import time

import pytest
from fastapi.testclient import TestClient

from backend.api.main import app
from scripts.init_db import init_database
from src.data.agnews import AGNewsAdapter
from src.data.db import DatabaseManager, normalize_db_url
from src.labeling.oracle import SimulatedOracle
from src.loop.orchestrator import ActiveLearningOrchestrator
from src.loop.state import LoopConfig
from src.models.text_net import TextMLP
from src.models.wrapper import ActiveLearningClassifier, seed_everything
from src.strategies.registry import get_strategy
from src.tracking.mlflow_tracker import MLflowTracker


def test_normalize_db_url_variants():
    assert normalize_db_url("sqlite:///foo.db") == "sqlite:///foo.db"
    pg1 = normalize_db_url("postgres://user:pass@localhost:5432/activepulse")
    assert pg1.startswith("postgresql+")
    pg2 = normalize_db_url("postgresql://user:pass@localhost:5432/activepulse")
    assert pg2.startswith("postgresql+")
    explicit = normalize_db_url("postgresql+pg8000://user:pass@localhost:5432/activepulse")
    assert explicit.startswith("postgresql+pg8000://")


def test_init_database_script(tmp_path):
    db_file = tmp_path / "init_check.db"
    resolved = init_database(f"sqlite:///{db_file}")
    assert os.path.exists(db_file)
    assert resolved.startswith("sqlite:///")


def test_mlflow_tracker_logs_params_metrics_and_artifacts(tmp_path):
    tracking_db = tmp_path / "mlflow.db"
    artifact_file = tmp_path / "sample_artifact.txt"
    artifact_file.write_text("activepulse test artifact", encoding="utf-8")

    tracker = MLflowTracker(
        tracking_uri=f"sqlite:///{tracking_db.as_posix()}",
        experiment_name="Test-ActivePulse-Suite",
        enabled=True,
    )
    assert tracker.is_active is True

    with tracker.start_run(
        run_name="run_seed_42",
        tags={"activepulse_experiment_id": "exp_mlflow_1"},
    ) as run_id:
        assert run_id is not None
        tracker.log_params({"dataset": "agnews", "strategy": "entropy", "seed": 42})
        tracker.log_round_metrics(
            round_num=0,
            accuracy=0.61,
            macro_f1=0.59,
            ece=0.05,
            n_labeled=20,
            n_pool=180,
            nll=0.92,
            alc=0.61,
            train_time_sec=0.45,
        )
        tracker.log_summary_metrics({"final_accuracy": 0.72, "alc": 0.67})
        tracker.log_artifact(str(artifact_file), artifact_path="reports")

    runs = tracker.compare_runs(experiment_ids=["exp_mlflow_1"])
    assert len(runs) == 1
    assert runs[0]["params.strategy"] == "entropy"
    assert float(runs[0]["metrics.final_accuracy"]) == pytest.approx(0.72)


def test_orchestrator_checkpoint_save_and_resume(tmp_path):
    db_file = tmp_path / "resume_orch.db"
    ckpt_dir = tmp_path / "ckpts"
    seed_everything(42)

    ds = AGNewsAdapter(max_samples=90, use_synthetic=True, seed=42)
    cfg = LoopConfig(
        dataset="agnews",
        model_type="text_mlp",
        strategy="least_confidence",
        initial_size=16,
        batch_size=8,
        max_rounds=1,
        max_epochs=2,
        seed=42,
        use_synthetic=True,
        max_samples=90,
        db_url=f"sqlite:///{db_file}",
    )

    model1 = ActiveLearningClassifier(TextMLP(64, 4), max_epochs=2, batch_size=8, device="cpu")
    orch1 = ActiveLearningOrchestrator(
        config=cfg,
        dataset=ds,
        model=model1,
        strategy=get_strategy("least_confidence", seed=42),
        labeler=SimulatedOracle(ds, seed=42),
        db=DatabaseManager(cfg.db_url, experiment_id="exp_resume"),
        checkpoint_dir=str(ckpt_dir),
    )
    orch1.run_simulation()
    assert os.path.exists(ckpt_dir / "latest.pt")
    assert os.path.exists(ckpt_dir / "rng_state.json")
    orch1.db.close()

    # Now resume from persisted round 1 to round 2
    cfg2 = LoopConfig(**{**cfg.__dict__, "max_rounds": 2})
    model2 = ActiveLearningClassifier(TextMLP(64, 4), max_epochs=2, batch_size=8, device="cpu")
    orch2 = ActiveLearningOrchestrator(
        config=cfg2,
        dataset=ds,
        model=model2,
        strategy=get_strategy("least_confidence", seed=42),
        labeler=SimulatedOracle(ds, seed=42),
        db=DatabaseManager(cfg.db_url, experiment_id="exp_resume"),
        checkpoint_dir=str(ckpt_dir),
    )
    orch2.run_simulation()
    hist = orch2.db.get_all_rounds_history()
    assert len(hist) == 3  # Round 0, 1, 2
    assert hist[-1]["round_num"] == 2
    orch2.db.close()


def test_experiment_jobs_evaluation_stop_resume_and_compare_api(tmp_path, monkeypatch):
    monkeypatch.setenv("ACTIVEPULSE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("ACTIVEPULSE_RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setenv("MLFLOW_ENABLED", "false")
    import backend.api.config as cfg

    monkeypatch.setattr(cfg, "DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(cfg, "RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setattr(cfg, "MLFLOW_ENABLED", False)

    client = TestClient(app)

    r_create = client.post(
        "/api/v1/experiments",
        json={
            "name": "Phase4 API Lifecycle",
            "dataset": "agnews",
            "model_type": "text_mlp",
            "strategy": "entropy",
            "mode": "simulated_oracle",
            "initial_size": 16,
            "batch_size": 8,
            "max_rounds": 2,
            "max_samples": 90,
            "use_synthetic": True,
            "max_epochs": 1,
            "auto_start": True,
        },
    )
    assert r_create.status_code == 200
    exp_id = r_create.json()["id"]

    # Wait for simulated run completion
    for _ in range(50):
        st = client.get(f"/api/v1/experiments/{exp_id}/progress").json()
        if st["status"] in ("completed", "failed"):
            break
        time.sleep(0.2)
    assert st["status"] == "completed"

    jobs = client.get(f"/api/v1/experiments/{exp_id}/jobs")
    assert jobs.status_code == 200
    assert len(jobs.json()["jobs"]) >= 1
    assert jobs.json()["jobs"][0]["status"] == "completed"

    ev = client.get(f"/api/v1/experiments/{exp_id}/evaluation")
    assert ev.status_code == 200
    ev_data = ev.json()
    assert ev_data["rounds_completed"] == 3
    assert ev_data["alc"] > 0.0
    assert ev_data["model_version"] is not None

    cmp_res = client.post("/api/v1/experiments/compare", json={"experiment_ids": [exp_id]})
    assert cmp_res.status_code == 200
    assert len(cmp_res.json()["experiments"]) == 1

    # Also test stop on created experiment before starting
    r_idle = client.post(
        "/api/v1/experiments",
        json={
            "name": "Stoppable Idle Run",
            "dataset": "agnews",
            "model_type": "text_mlp",
            "strategy": "random",
            "mode": "simulated_oracle",
            "initial_size": 16,
            "batch_size": 8,
            "max_rounds": 1,
            "max_samples": 90,
            "use_synthetic": True,
            "max_epochs": 1,
            "auto_start": False,
        },
    )
    idle_id = r_idle.json()["id"]
    stopped = client.post(f"/api/v1/experiments/{idle_id}/stop")
    assert stopped.status_code == 200
    assert stopped.json()["status"] in ("paused", "cancelled")

    resumed = client.post(f"/api/v1/experiments/{idle_id}/resume")
    assert resumed.status_code == 200
    for _ in range(40):
        st2 = client.get(f"/api/v1/experiments/{idle_id}/progress").json()
        if st2["status"] in ("completed", "failed"):
            break
        time.sleep(0.2)
    assert st2["status"] == "completed"

    client.delete(f"/api/v1/experiments/{exp_id}")
    client.delete(f"/api/v1/experiments/{idle_id}")

