"""Integration tests for ActivePulse Studio FastAPI endpoints."""

import os
import tempfile
import time
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def api_client():
    tmp_dir = tempfile.mkdtemp(prefix="activepulse_api_test_")
    os.environ["ACTIVEPULSE_DATA_DIR"] = tmp_dir
    from backend.api.main import create_app

    app = create_app()
    with TestClient(app) as client:
        yield client


def test_health_and_catalog(api_client):
    r = api_client.get("/api/v1/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["registered_strategies"] >= 9

    ds = api_client.get("/api/v1/datasets")
    assert ds.status_code == 200
    assert {d["id"] for d in ds.json()} == {"cifar10", "agnews"}

    models = api_client.get("/api/v1/models")
    assert models.status_code == 200
    assert {m["id"] for m in models.json()} == {"cifar_cnn", "resnet18", "text_mlp"}

    strats = api_client.get("/api/v1/strategies")
    assert strats.status_code == 200
    assert any(s["id"] == "hybrid" for s in strats.json())


def test_dataset_stats_and_samples(api_client):
    r_stats = api_client.get("/api/v1/datasets/cifar10/stats?max_samples=100&use_synthetic=true")
    assert r_stats.status_code == 200
    stats = r_stats.json()
    assert stats["split_stats"]["n_total"] == 100
    assert stats["split_stats"]["n_test"] == 20
    assert len(stats["class_distribution"]) == 10

    r_samples = api_client.get("/api/v1/datasets/cifar10/samples?max_samples=100&limit=5&use_synthetic=true")
    assert r_samples.status_code == 200
    samples = r_samples.json()["samples"]
    assert len(samples) == 5
    assert samples[0]["image_base64"].startswith("data:image/png;base64,")

    r_ag = api_client.get("/api/v1/datasets/agnews/samples?max_samples=80&limit=3&use_synthetic=true")
    assert r_ag.status_code == 200
    assert r_ag.json()["samples"][0]["feature_summary"] is not None


def test_simulated_experiment_lifecycle(api_client):
    payload = {
        "name": "API Smoke Sim",
        "dataset": "agnews",
        "model_type": "text_mlp",
        "strategy": "entropy",
        "mode": "simulated_oracle",
        "initial_size": 16,
        "batch_size": 8,
        "max_rounds": 2,
        "max_samples": 100,
        "use_synthetic": True,
        "max_epochs": 2,
        "lr": 0.02,
        "seed": 42,
        "auto_start": True,
    }
    res = api_client.post("/api/v1/experiments", json=payload)
    assert res.status_code == 200
    exp_id = res.json()["id"]

    # Poll until completed
    for _ in range(60):
        prog = api_client.get(f"/api/v1/experiments/{exp_id}/progress").json()
        if prog["status"] in ("completed", "failed"):
            break
        time.sleep(0.3)

    assert prog["status"] == "completed", f"Unexpected error: {prog.get('error_message')}"

    detail = api_client.get(f"/api/v1/experiments/{exp_id}").json()
    assert len(detail["rounds"]) == 3  # Round 0, 1, 2
    assert detail["cost_savings"] is not None

    pca = api_client.get(f"/api/v1/experiments/{exp_id}/pca").json()
    assert len(pca["points"]) > 0

    sel = api_client.get(f"/api/v1/experiments/{exp_id}/selected-samples").json()
    assert len(sel["samples"]) == 8

    csv_res = api_client.get(f"/api/v1/experiments/{exp_id}/export")
    assert csv_res.status_code == 200
    assert "test_acc" in csv_res.text

    del_res = api_client.delete(f"/api/v1/experiments/{exp_id}")
    assert del_res.status_code == 200


def test_hitl_annotation_workflow_hides_ground_truth(api_client):
    payload = {
        "name": "HITL Smoke Test",
        "dataset": "cifar10",
        "model_type": "cifar_cnn",
        "strategy": "hybrid",
        "mode": "human_in_the_loop",
        "initial_size": 16,
        "batch_size": 3,
        "max_rounds": 2,
        "max_samples": 80,
        "use_synthetic": True,
        "max_epochs": 2,
        "lr": 0.01,
        "seed": 7,
        "auto_start": True,
    }
    res = api_client.post("/api/v1/experiments", json=payload)
    assert res.status_code == 200
    exp_id = res.json()["id"]

    for _ in range(60):
        prog = api_client.get(f"/api/v1/experiments/{exp_id}/progress").json()
        if prog["status"] in ("awaiting_annotations", "failed"):
            break
        time.sleep(0.3)

    assert prog["status"] == "awaiting_annotations", f"Error: {prog.get('error_message')}"

    pending = api_client.get(f"/api/v1/annotations/pending?experiment_id={exp_id}").json()
    assert pending["pending_count"] == 3
    for s in pending["samples"]:
        assert "true_label" not in s
        assert "true_class_id" not in s
        assert "true_class_name" not in s
        assert s["image_base64"].startswith("data:image/png;base64,")

    # Submit labels for all 3 queued samples
    for s in pending["samples"]:
        sub = api_client.post(
            "/api/v1/annotations/submit",
            json={
                "experiment_id": exp_id,
                "sample_id": s["sample_id"],
                "label_id": s["predicted_class_id"] or 0,
                "annotator": "human_reviewer",
            },
        )
        assert sub.status_code == 200

    hist = api_client.get(f"/api/v1/annotations/history?experiment_id={exp_id}").json()
    assert hist["human_count"] == 3
    assert hist["total_count"] == 19  # 16 cold_start + 3 human

    # Advance round (retrain + queue round 2)
    adv = api_client.post(f"/api/v1/experiments/{exp_id}/advance")
    assert adv.status_code == 200

    for _ in range(60):
        prog = api_client.get(f"/api/v1/experiments/{exp_id}/progress").json()
        if prog["status"] in ("awaiting_annotations", "completed", "failed"):
            break
        time.sleep(0.3)

    assert prog["status"] in ("awaiting_annotations", "completed")
    api_client.delete(f"/api/v1/experiments/{exp_id}")


def test_benchmarks_and_label_studio_status(api_client):
    res = api_client.get("/api/v1/benchmarks/results")
    assert res.status_code == 200

    ls = api_client.get("/api/v1/webhooks/label-studio/status")
    assert ls.status_code == 200
    assert ls.json()["integration_status"] in ("unconfigured", "configured", "connected", "ready")

    sav = api_client.get("/api/v1/benchmarks/savings")
    assert sav.status_code == 200
    cur = api_client.get("/api/v1/benchmarks/curves")
    assert cur.status_code == 200

