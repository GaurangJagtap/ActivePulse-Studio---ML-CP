"""Phase 3 tests: Label Studio SDK integration, webhooks, idempotency, and Cohen's Kappa QA."""
from __future__ import annotations

import hashlib
import hmac
import json
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from backend.api.main import app
from backend.api.services.dataset_service import get_cached_dataset
from src.data.db import DatabaseManager
from src.labeling.label_studio import (
    LabelStudioConnectionConfig,
    LabelStudioLabeler,
    LabelStudioManager,
    build_labeling_config,
    generate_model_pre_annotations,
    tensor_to_png_data_uri,
)
from src.labeling.quality import (
    compute_cohens_kappa,
    evaluate_annotation_quality,
    select_double_annotation_samples,
)
from src.models.cnn import CIFARConvNet
from src.models.wrapper import ActiveLearningClassifier


class _FakeProjectsResource:
    def __init__(self):
        self._items = []
        self._next_id = 10

    def list(self):
        return list(self._items)

    def create(self, **kwargs):
        proj = {"id": self._next_id, **kwargs}
        self._next_id += 1
        self._items.append(proj)
        return proj


class _FakeTasksResource:
    def __init__(self):
        self._items = []
        self._next_id = 100

    def create(self, **kwargs):
        tid = self._next_id
        self._next_id += 1
        task = {"id": tid, "annotations": [], **kwargs}
        self._items.append(task)
        return task

    def list(self, project=None):
        if project is None:
            return list(self._items)
        return [t for t in self._items if t.get("project") == project]


class _FakePredictionsResource:
    def __init__(self):
        self.created = []

    def create(self, **kwargs):
        pred = {"id": len(self.created) + 1, **kwargs}
        self.created.append(pred)
        return pred


class _FakeWebhooksResource:
    def __init__(self):
        self.created = []

    def create(self, **kwargs):
        wh = {"id": len(self.created) + 1, **kwargs}
        self.created.append(wh)
        return wh


class _FakeLSClient:
    def __init__(self):
        self.projects = _FakeProjectsResource()
        self.tasks = _FakeTasksResource()
        self.predictions = _FakePredictionsResource()
        self.webhooks = _FakeWebhooksResource()


def test_xml_configs_and_png_encoding():
    img_cfg = build_labeling_config("image", ["cat", "dog", "bird & plane"])
    assert "<Image name=\"image\"" in img_cfg
    assert "bird &amp; plane" in img_cfg

    txt_cfg = build_labeling_config("text", ["World", "Sports"])
    assert "<Text name=\"text\"" in txt_cfg
    assert '<Choice value="Sports"/>' in txt_cfg

    uri = tensor_to_png_data_uri(torch.rand(3, 32, 32))
    assert uri.startswith("data:image/png;base64,")


def test_cohens_kappa_and_quality_evaluation():
    chosen = select_double_annotation_samples([f"s{i}" for i in range(20)], overlap_rate=0.10, seed=42)
    assert len(chosen) == 2

    kappa_small, note_small = compute_cohens_kappa([0], [0])
    assert kappa_small is None
    assert "Insufficient" in note_small

    kappa_degen, note_degen = compute_cohens_kappa([1, 1, 1], [1, 1, 1])
    assert kappa_degen is None
    assert "same class" in note_degen

    kappa_perfect, note_ok = compute_cohens_kappa([0, 1, 0, 1], [0, 1, 0, 1])
    assert kappa_perfect == pytest.approx(1.0)
    assert note_ok is None

    # Disagreement detection + pairwise stats strictly between human annotators
    records = [
        {"sample_id": "s0", "label_id": 0, "annotator": "human_alice", "source_type": "human"},
        {"sample_id": "s0", "label_id": 0, "annotator": "human_bob", "source_type": "human"},
        {"sample_id": "s1", "label_id": 1, "annotator": "human_alice", "source_type": "human"},
        {"sample_id": "s1", "label_id": 2, "annotator": "human_bob", "source_type": "human"},
        {"sample_id": "s2", "label_id": 2, "annotator": "human_alice", "source_type": "human"},
        {"sample_id": "s2", "label_id": 2, "annotator": "human_bob", "source_type": "human"},
        {"sample_id": "s3", "label_id": 1, "annotator": "human_alice", "source_type": "human"},
    ]
    report = evaluate_annotation_quality(records, class_names=["airplane", "auto", "bird"])
    assert report["overlapping_samples"] == 3
    assert report["agreed_overlapping_samples"] == 2
    assert report["disagreement_count"] == 1
    assert report["disagreements"][0]["sample_id"] == "s1"
    assert report["disagreements"][0]["needs_review"] is True
    assert report["cohens_kappa"] is not None
    assert len(report["pairwise_kappa"]) == 1
    assert len(report["annotator_stats"]) == 2


def test_label_studio_pre_annotations_and_idempotent_sync(tmp_path):
    dataset = get_cached_dataset("cifar10", max_samples=60, use_synthetic=True, seed=42)
    db_path = tmp_path / "ls_test.db"
    db = DatabaseManager(f"sqlite:///{db_path}", experiment_id="exp_ls")

    ids = dataset.get_all_sample_ids()
    init_ids = ids[:16]
    pool_ids = ids[16:40]
    db.initialize_samples(
        sample_ids=init_ids + pool_ids,
        splits=["pool"] * (len(init_ids) + len(pool_ids)),
        true_labels=dataset.get_ground_truth(init_ids + pool_ids),
    )
    db.mark_labeled(init_ids, dataset.get_ground_truth(init_ids), round_num=0, annotator="cold_start")

    clf = ActiveLearningClassifier(
        CIFARConvNet(num_classes=10),
        max_epochs=1,
        batch_size=8,
        device="cpu",
    )
    clf.fit(dataset.get_features(init_ids), np.array(dataset.get_ground_truth(init_ids)))

    fake_ls = _FakeLSClient()
    manager = LabelStudioManager(
        config=LabelStudioConnectionConfig(base_url="http://localhost:8080", api_key="test-key"),
        client=fake_ls,
    )
    labeler = LabelStudioLabeler(
        db=db,
        dataset=dataset,
        manager=manager,
        model_wrapper=clf,
        double_annotation_rate=0.25,
    )

    queried = pool_ids[:4]
    db.mark_queued(queried)
    res = labeler.queue_and_upload_batch(queried, round_num=1, strategy="entropy", seed=42)
    assert res["tasks_created"] == 4
    assert len(fake_ls.predictions.created) == 4

    # Pre-annotations must NEVER convert queued samples to labeled
    assert len(db.get_queued_sample_ids()) == 4

    # Complete task #1 in fake Label Studio (annotator accepts/overrides choice)
    task0 = fake_ls.tasks._items[0]
    task0["annotations"] = [
        {
            "id": 9001,
            "completed_by": {"username": "alice"},
            "was_cancelled": False,
            "result": [{"value": {"choices": [dataset.class_names[2]]}}],
        }
    ]
    sync1 = manager.sync_completed_annotations(
        project_id=res["project_id"],
        class_names=dataset.class_names,
        db=db,
        round_num=1,
        batch_id="exp_ls_r1",
    )
    assert sync1["synced_count"] == 1
    assert db.get_sample(queried[0]).label == 2
    assert db.get_sample(queried[0]).status == "labeled"

    # Re-syncing the same annotation ID must skip idempotently
    sync2 = manager.sync_completed_annotations(
        project_id=res["project_id"],
        class_names=dataset.class_names,
        db=db,
        round_num=1,
        batch_id="exp_ls_r1",
    )
    assert sync2["synced_count"] == 0
    assert sync2["duplicate_skipped"] == 1
    db.close()


def test_webhook_secret_idempotency_and_quality_api(tmp_path, monkeypatch):
    monkeypatch.setenv("ACTIVEPULSE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("ACTIVEPULSE_RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setenv("LABEL_STUDIO_WEBHOOK_SECRET", "top-secret-webhook")
    import backend.api.config as cfg

    monkeypatch.setattr(cfg, "DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(cfg, "RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setattr(cfg, "LABEL_STUDIO_WEBHOOK_SECRET", "top-secret-webhook")

    client = TestClient(app)

    # Create HITL experiment with 1-round small pool
    create_resp = client.post(
        "/api/v1/experiments",
        json={
            "name": "Phase3 Webhook QA Test",
            "dataset": "agnews",
            "model_type": "text_mlp",
            "strategy": "margin",
            "mode": "human_in_the_loop",
            "initial_size": 16,
            "batch_size": 2,
            "max_rounds": 2,
            "max_samples": 70,
            "use_synthetic": True,
            "max_epochs": 1,
            "auto_start": True,
        },
    )
    assert create_resp.status_code == 200
    exp_id = create_resp.json()["id"]

    # Wait briefly for awaiting_annotations
    import time

    for _ in range(40):
        st = client.get(f"/api/v1/experiments/{exp_id}/progress").json()
        if st["status"] in ("awaiting_annotations", "completed", "failed"):
            break
        time.sleep(0.25)

    pending = client.get(f"/api/v1/annotations/pending?experiment_id={exp_id}").json()
    assert pending["pending_count"] == 2
    s0 = pending["samples"][0]["sample_id"]
    s1 = pending["samples"][1]["sample_id"]

    # Missing secret -> 401
    unauth = client.post(
        "/api/v1/webhooks/label-studio",
        json={"action": "ANNOTATION_CREATED", "experiment_id": exp_id, "sample_id": s0, "label_id": 1},
    )
    assert unauth.status_code == 401

    # Valid secret + duplicate webhook event idempotency check
    payload = {
        "action": "ANNOTATION_CREATED",
        "experiment_id": exp_id,
        "sample_id": s0,
        "label_id": 1,
        "annotator": "alice",
        "webhook_event_id": "evt_101",
    }
    ok1 = client.post(
        "/api/v1/webhooks/label-studio",
        json=payload,
        headers={"X-Label-Studio-Secret": "top-secret-webhook"},
    )
    assert ok1.status_code == 200
    assert ok1.json()["status"] == "ingested"

    ok_dup = client.post(
        "/api/v1/webhooks/label-studio",
        json=payload,
        headers={"X-Label-Studio-Secret": "top-secret-webhook"},
    )
    assert ok_dup.status_code == 200
    assert ok_dup.json()["duplicate_skipped"] is True

    # Double-annotate s0 with a second independent human annotator (disagreement!) via HMAC header
    raw_body_dict = {
        "action": "ANNOTATION_CREATED",
        "experiment_id": exp_id,
        "sample_id": s0,
        "label_id": 2,
        "annotator": "bob",
        "webhook_event_id": "evt_102",
    }
    raw_bytes = json.dumps(raw_body_dict).encode("utf-8")
    sig = hmac.new(b"top-secret-webhook", raw_bytes, hashlib.sha256).hexdigest()
    ok2 = client.post(
        "/api/v1/webhooks/label-studio",
        content=raw_bytes,
        headers={"Content-Type": "application/json", "X-Label-Studio-Signature": f"sha256={sig}"},
    )
    assert ok2.status_code == 200

    # Annotate s1 by both alice & bob in agreement
    for idx, (ann, evt) in enumerate([("alice", "evt_103"), ("bob", "evt_104")]):
        r = client.post(
            "/api/v1/webhooks/label-studio",
            json={
                "action": "ANNOTATION_CREATED",
                "experiment_id": exp_id,
                "sample_id": s1,
                "label_id": 0,
                "annotator": ann,
                "webhook_event_id": evt,
            },
            headers={"X-Label-Studio-Secret": "top-secret-webhook"},
        )
        assert r.status_code == 200

    # Check QA endpoint & completed annotations endpoint
    qa = client.get(f"/api/v1/annotations/quality?experiment_id={exp_id}")
    assert qa.status_code == 200
    qa_data = qa.json()
    assert qa_data["overlapping_samples"] == 2
    assert qa_data["disagreement_count"] == 1
    assert qa_data["cohens_kappa"] is not None

    comp = client.get(f"/api/v1/annotations/completed?experiment_id={exp_id}")
    assert comp.status_code == 200
    assert comp.json()["total_completed"] == 4

    client.delete(f"/api/v1/experiments/{exp_id}")
