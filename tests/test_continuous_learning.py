"""End-to-end tests for Continuous Learning and Automatic Self-Retraining."""

import time
import pytest
from fastapi.testclient import TestClient

from backend.api.main import create_app


@pytest.fixture
def client():
    app = create_app()
    return TestClient(app)


def test_continuous_learning_and_automatic_self_retraining(client, tmp_path, monkeypatch):
    monkeypatch.setenv("ACTIVEPULSE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("ACTIVEPULSE_RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setenv("MLFLOW_ENABLED", "false")

    # 1. Bootstrap a Human-in-the-Loop experiment
    res_exp = client.post(
        "/api/v1/experiments",
        json={
            "name": "Continuous Learning End-to-End Test",
            "dataset": "agnews",
            "model_type": "text_mlp",
            "strategy": "entropy",
            "mode": "human_in_the_loop",
            "initial_size": 16,
            "batch_size": 4,
            "max_rounds": 2,
            "max_samples": 80,
            "use_synthetic": True,
            "max_epochs": 2,
            "lr": 0.01,
            "seed": 42,
            "auto_start": True,
        },
    )
    assert res_exp.status_code == 200
    exp_id = res_exp.json()["id"]

    # Wait for bootstrap to queue samples in Annotation Studio
    for _ in range(60):
        prog = client.get(f"/api/v1/experiments/{exp_id}/progress").json()
        if prog["status"] in ("awaiting_annotations", "failed"):
            break
        time.sleep(0.3)

    assert prog["status"] == "awaiting_annotations", f"Bootstrap failed: {prog.get('error_message')}"

    # 2. Check pending samples and model predictions
    res_pending = client.get(f"/api/v1/annotations/pending?experiment_id={exp_id}")
    assert res_pending.status_code == 200
    pending_data = res_pending.json()
    assert pending_data["pending_count"] == 4
    samples = pending_data["samples"]
    first_sample = samples[0]

    model_pred = first_sample.get("predicted_class_id") or 0
    # Create a genuine human correction by submitting a different label
    corrected_label_id = (model_pred + 1) % len(first_sample["class_names"])

    # 3. Submit human correction
    res_sub = client.post(
        "/api/v1/annotations/submit",
        json={
            "experiment_id": exp_id,
            "sample_id": first_sample["sample_id"],
            "label_id": corrected_label_id,
            "annotator": "expert_verifier",
        },
    )
    assert res_sub.status_code == 200
    sub_data = res_sub.json()
    assert sub_data["is_correction"] is True
    assert sub_data["continuous_learning_triggered"] is True

    # 4. Wait for background continuous retraining to execute and evaluate candidate
    for _ in range(60):
        cl_res = client.get(f"/api/v1/continuous-learning/status?experiment_id={exp_id}").json()
        if cl_res["retraining_status"] in (
            "model_updated_successfully",
            "candidate_rejected",
            "retraining_failed",
        ):
            break
        time.sleep(0.4)

    assert cl_res["retraining_status"] in ("model_updated_successfully", "candidate_rejected")
    assert cl_res["total_human_verified"] >= 1
    assert cl_res["newly_corrected"] >= 1
    assert len(cl_res["history"]) >= 1

    latest_event = cl_res["history"][0]
    assert latest_event["status"] in ("promoted", "rejected")
    assert latest_event["samples_incorporated"] >= 1
    assert latest_event["candidate_metrics"]["accuracy"] is not None

    # 5. Test manual retraining trigger endpoint
    res_manual = client.post(
        "/api/v1/continuous-learning/retrain",
        json={"experiment_id": exp_id, "force": True},
    )
    assert res_manual.status_code == 200
    assert res_manual.json()["status"] == "triggered"

    client.delete(f"/api/v1/experiments/{exp_id}")

