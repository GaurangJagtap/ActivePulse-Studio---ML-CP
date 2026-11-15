"""Database persistence layer supporting SQLite and PostgreSQL for ActivePulse Studio."""

from datetime import datetime, timezone
import importlib.util
import json
from typing import Any, Dict, List, Optional, Tuple
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    inspect,
    select,
    text,
    update,
)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, declarative_base, sessionmaker

Base = declarative_base()


def _utcnow() -> datetime:
    """Return timezone-naive UTC datetime without triggering Python 3.12+ utcnow warnings."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def normalize_db_url(db_url: str) -> str:
    """Normalize PostgreSQL URLs to pick an installed SQLAlchemy DBAPI driver."""
    if db_url.startswith("postgres://"):
        db_url = "postgresql://" + db_url[len("postgres://") :]

    if db_url.startswith("postgresql://"):
        if importlib.util.find_spec("psycopg2") is not None:
            return db_url.replace("postgresql://", "postgresql+psycopg2://", 1)
        if importlib.util.find_spec("psycopg") is not None:
            return db_url.replace("postgresql://", "postgresql+psycopg://", 1)
        if importlib.util.find_spec("pg8000") is not None:
            return db_url.replace("postgresql://", "postgresql+pg8000://", 1)
    return db_url


class ExperimentRecord(Base):
    """Persisted experiment metadata, configuration, and external integrations."""

    __tablename__ = "experiments"

    id = Column(String(64), primary_key=True, index=True)
    name = Column(String(160), nullable=False)
    dataset = Column(String(64), nullable=False)
    model_type = Column(String(64), nullable=False)
    strategy = Column(String(64), nullable=False)
    mode = Column(String(64), nullable=False, default="simulated_oracle")
    status = Column(String(64), nullable=False, default="created", index=True)
    current_round = Column(Integer, nullable=False, default=0)
    max_rounds = Column(Integer, nullable=False, default=3)
    current_stage = Column(String(128), nullable=False, default="Initialized")
    stage_index = Column(Integer, nullable=False, default=0)
    config_json = Column(Text, nullable=True)
    stop_reason = Column(Text, nullable=True)
    mlflow_run_id = Column(String(128), nullable=True)
    label_studio_project_id = Column(Integer, nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=_utcnow)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)


class SampleRecord(Base):
    """SQLAlchemy model for dataset sample tracking across active learning rounds."""

    __tablename__ = "samples"
    __table_args__ = (
        UniqueConstraint("experiment_id", "sample_id", name="uq_samples_exp_sample"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    experiment_id = Column(String(64), nullable=False, default="default", index=True)
    sample_id = Column(String(64), nullable=False, index=True)
    split = Column(String(32), nullable=False, index=True)  # 'pool', 'val', 'test'
    status = Column(String(32), nullable=False, default="unlabeled", index=True)  # 'unlabeled', 'queued', 'labeled'
    label = Column(Integer, nullable=True)
    true_label = Column(Integer, nullable=False)
    round_added = Column(Integer, nullable=True)
    annotator = Column(String(64), nullable=True)
    label_studio_task_id = Column(Integer, nullable=True, index=True)
    pre_annotation_label = Column(Integer, nullable=True)
    pre_annotation_confidence = Column(Float, nullable=True)
    pre_annotation_probs_json = Column(Text, nullable=True)
    needs_review = Column(Boolean, nullable=False, default=False)
    batch_id = Column(String(64), nullable=True, index=True)
    created_at = Column(DateTime, default=_utcnow)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)


class RoundRecord(Base):
    """SQLAlchemy model for recording round metrics."""

    __tablename__ = "rounds"
    __table_args__ = (
        UniqueConstraint("experiment_id", "round_num", name="uq_rounds_exp_num"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    experiment_id = Column(String(64), nullable=False, default="default", index=True)
    round_num = Column(Integer, nullable=False)
    strategy = Column(String(64), nullable=False)
    n_labeled = Column(Integer, nullable=False)
    n_pool = Column(Integer, nullable=False)
    test_acc = Column(Float, nullable=False)
    test_f1 = Column(Float, nullable=False)
    ece = Column(Float, nullable=True)
    temperature = Column(Float, nullable=True)
    nll = Column(Float, nullable=True)
    train_time_sec = Column(Float, nullable=True)
    timestamp = Column(DateTime, default=_utcnow)


class AnnotationBatchRecord(Base):
    """Tracks queued HITL annotation batches and completion thresholds."""

    __tablename__ = "annotation_batches"

    batch_id = Column(String(64), primary_key=True, index=True)
    experiment_id = Column(String(64), nullable=False, default="default", index=True)
    round_num = Column(Integer, nullable=False)
    strategy = Column(String(64), nullable=False)
    total_tasks = Column(Integer, nullable=False, default=0)
    completed_tasks = Column(Integer, nullable=False, default=0)
    threshold_count = Column(Integer, nullable=False, default=0)
    double_annotation_rate = Column(Float, nullable=False, default=0.10)
    status = Column(String(32), nullable=False, default="pending")  # pending, ready_to_retrain, completed
    created_at = Column(DateTime, default=_utcnow)
    completed_at = Column(DateTime, nullable=True)


class AnnotationRecord(Base):
    """Individual human / Label Studio / oracle annotation events (supports multi-annotator & webhook idempotency)."""

    __tablename__ = "annotations"
    __table_args__ = (
        UniqueConstraint("experiment_id", "sample_id", "annotator", "round_num", name="uq_ann_exp_samp_user_round"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    experiment_id = Column(String(64), nullable=False, default="default", index=True)
    sample_id = Column(String(64), nullable=False, index=True)
    batch_id = Column(String(64), nullable=True, index=True)
    round_num = Column(Integer, nullable=False, default=0)
    annotator = Column(String(64), nullable=False)
    label_id = Column(Integer, nullable=False)
    source_type = Column(String(32), nullable=False, default="human")
    is_double_annotation = Column(Boolean, nullable=False, default=False)
    webhook_event_id = Column(String(128), nullable=True, unique=True, index=True)
    label_studio_annotation_id = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=_utcnow)


class QualityMetricRecord(Base):
    """Stores Cohen's Kappa and inter-annotator agreement snapshots per round."""

    __tablename__ = "quality_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    experiment_id = Column(String(64), nullable=False, default="default", index=True)
    round_num = Column(Integer, nullable=False, default=0)
    n_overlapping = Column(Integer, nullable=False, default=0)
    cohens_kappa = Column(Float, nullable=True)
    raw_agreement = Column(Float, nullable=True)
    disagreement_count = Column(Integer, nullable=False, default=0)
    flagged_sample_ids_json = Column(Text, nullable=True)
    per_class_json = Column(Text, nullable=True)
    annotator_stats_json = Column(Text, nullable=True)
    computed_at = Column(DateTime, default=_utcnow)


class ExperimentJobRecord(Base):
    """Background worker execution job state."""

    __tablename__ = "experiment_jobs"

    job_id = Column(String(64), primary_key=True, index=True)
    experiment_id = Column(String(64), nullable=False, index=True)
    job_type = Column(String(64), nullable=False)  # simulated_oracle, hitl_bootstrap, hitl_advance, resume
    status = Column(String(32), nullable=False, default="queued", index=True)  # queued, running, completed, failed, cancelled
    progress_pct = Column(Float, nullable=False, default=0.0)
    current_stage = Column(String(128), nullable=True)
    error_message = Column(Text, nullable=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=_utcnow)


class ModelVersionRecord(Base):
    """Persisted model checkpoints, calibration values, and MLflow run linkage."""

    __tablename__ = "model_versions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    experiment_id = Column(String(64), nullable=False, index=True)
    round_num = Column(Integer, nullable=False)
    model_type = Column(String(64), nullable=False)
    checkpoint_path = Column(Text, nullable=False)
    temperature = Column(Float, nullable=True)
    val_ece = Column(Float, nullable=True)
    test_acc = Column(Float, nullable=True)
    test_f1 = Column(Float, nullable=True)
    test_ece = Column(Float, nullable=True)
    test_nll = Column(Float, nullable=True)
    train_time_sec = Column(Float, nullable=True)
    mlflow_run_id = Column(String(128), nullable=True)
    created_at = Column(DateTime, default=_utcnow)


class ContinuousFeedbackRecord(Base):
    """Tracks human corrections and confirmations for continuous self-retraining."""

    __tablename__ = "continuous_feedback"

    id = Column(Integer, primary_key=True, autoincrement=True)
    experiment_id = Column(String(64), nullable=False, index=True)
    sample_id = Column(String(64), nullable=False, index=True)
    original_pred_id = Column(Integer, nullable=True)
    original_pred_name = Column(String(64), nullable=True)
    original_confidence = Column(Float, nullable=True)
    human_label_id = Column(Integer, nullable=False)
    human_label_name = Column(String(64), nullable=False)
    is_correction = Column(Boolean, nullable=False, default=False)
    annotator = Column(String(64), nullable=False)
    model_version_id = Column(String(64), nullable=True)
    used_in_training = Column(Boolean, nullable=False, default=False)
    retrain_job_id = Column(String(64), nullable=True)
    created_at = Column(DateTime, default=_utcnow)


class ContinuousRetrainEventRecord(Base):
    """Audits each continuous retraining cycle, candidate comparison, and promotion decision."""

    __tablename__ = "continuous_retrain_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_id = Column(String(64), primary_key=False, unique=True, index=True)
    experiment_id = Column(String(64), nullable=False, index=True)
    trigger_type = Column(String(32), nullable=False, default="auto")  # auto, micro_batch, manual
    trigger_sample_id = Column(String(64), nullable=True)
    samples_incorporated = Column(Integer, nullable=False, default=0)
    total_training_samples = Column(Integer, nullable=False, default=0)
    prev_model_version = Column(String(64), nullable=True)
    candidate_model_version = Column(String(64), nullable=True)
    prev_accuracy = Column(Float, nullable=True)
    prev_macro_f1 = Column(Float, nullable=True)
    prev_ece = Column(Float, nullable=True)
    candidate_accuracy = Column(Float, nullable=True)
    candidate_macro_f1 = Column(Float, nullable=True)
    candidate_ece = Column(Float, nullable=True)
    status = Column(String(32), nullable=False, default="pending")  # pending, training, evaluating, promoted, rejected, failed
    decision_reason = Column(Text, nullable=True)
    created_at = Column(DateTime, default=_utcnow)
    completed_at = Column(DateTime, nullable=True)


class DatabaseManager:
    """Manages SQLite / PostgreSQL connection and transactional queries for AL state."""

    def __init__(self, db_url: str = "sqlite:///al_platform.db", experiment_id: str = "default"):
        self.raw_db_url = db_url
        self.db_url = normalize_db_url(db_url)
        self.experiment_id = experiment_id

        engine_kwargs: Dict[str, Any] = {"echo": False}
        if self.db_url.startswith("sqlite"):
            engine_kwargs["connect_args"] = {"check_same_thread": False}
        else:
            engine_kwargs["pool_pre_ping"] = True
            engine_kwargs["pool_size"] = 5
            engine_kwargs["max_overflow"] = 10

        self.engine = create_engine(self.db_url, **engine_kwargs)
        Base.metadata.create_all(self.engine)
        self._migrate_existing_columns()
        self.SessionLocal = sessionmaker(bind=self.engine, autoflush=False, autocommit=False)

    def _migrate_existing_columns(self) -> None:
        """Non-destructively add new columns to legacy SQLite/Postgres tables when upgrading."""
        try:
            insp = inspect(self.engine)
            is_sqlite = self.db_url.startswith("sqlite")
            bool_type = "INTEGER" if is_sqlite else "BOOLEAN"
            extra_columns = {
                "samples": {
                    "experiment_id": "VARCHAR(64) DEFAULT 'default'",
                    "label_studio_task_id": "INTEGER",
                    "pre_annotation_label": "INTEGER",
                    "pre_annotation_confidence": "FLOAT",
                    "pre_annotation_probs_json": "TEXT",
                    "needs_review": f"{bool_type} DEFAULT 0",
                    "batch_id": "VARCHAR(64)",
                },
                "rounds": {
                    "experiment_id": "VARCHAR(64) DEFAULT 'default'",
                    "nll": "FLOAT",
                    "train_time_sec": "FLOAT",
                },
            }
            with self.engine.begin() as conn:
                for table_name, cols in extra_columns.items():
                    if not insp.has_table(table_name):
                        continue
                    existing = {c["name"] for c in insp.get_columns(table_name)}
                    for col_name, col_def in cols.items():
                        if col_name not in existing:
                            conn.execute(
                                text(f"ALTER TABLE {table_name} ADD COLUMN {col_name} {col_def}")
                            )
        except Exception:
            # Never break initialization on legacy inspect edge cases
            pass

    def get_session(self) -> Session:
        return self.SessionLocal()

    def initialize_samples(
        self,
        sample_ids: List[str],
        splits: List[str],
        true_labels: List[int],
    ) -> None:
        """Populate database with sample metadata for this experiment if not already present."""
        with self.get_session() as session:
            existing_count = (
                session.query(SampleRecord)
                .filter(SampleRecord.experiment_id == self.experiment_id)
                .count()
            )
            if existing_count > 0:
                return

            records = [
                SampleRecord(
                    experiment_id=self.experiment_id,
                    sample_id=sid,
                    split=split,
                    status="unlabeled" if split == "pool" else "labeled",
                    label=t_lbl if split != "pool" else None,
                    true_label=t_lbl,
                    round_added=0 if split != "pool" else None,
                    annotator="pre_split" if split != "pool" else None,
                )
                for sid, split, t_lbl in zip(sample_ids, splits, true_labels)
            ]
            session.bulk_save_objects(records)
            session.commit()

    def get_sample(self, sample_id: str) -> Optional[SampleRecord]:
        with self.get_session() as session:
            return (
                session.query(SampleRecord)
                .filter(
                    SampleRecord.experiment_id == self.experiment_id,
                    SampleRecord.sample_id == sample_id,
                )
                .first()
            )

    def get_sample_by_ls_task_id(self, task_id: int) -> Optional[SampleRecord]:
        with self.get_session() as session:
            return (
                session.query(SampleRecord)
                .filter(
                    SampleRecord.experiment_id == self.experiment_id,
                    SampleRecord.label_studio_task_id == task_id,
                )
                .first()
            )

    def get_pool_sample_ids(self) -> List[str]:
        """Return all sample_ids currently in the unlabeled pool."""
        with self.get_session() as session:
            stmt = (
                select(SampleRecord.sample_id)
                .where(
                    SampleRecord.experiment_id == self.experiment_id,
                    SampleRecord.split == "pool",
                    SampleRecord.status == "unlabeled",
                )
                .order_by(SampleRecord.sample_id)
            )
            return [row[0] for row in session.execute(stmt).all()]

    def get_queued_sample_ids(self) -> List[str]:
        """Return all sample_ids currently queued for labeling."""
        with self.get_session() as session:
            stmt = (
                select(SampleRecord.sample_id)
                .where(
                    SampleRecord.experiment_id == self.experiment_id,
                    SampleRecord.split == "pool",
                    SampleRecord.status == "queued",
                )
                .order_by(SampleRecord.sample_id)
            )
            return [row[0] for row in session.execute(stmt).all()]

    def get_labeled_samples(self) -> Tuple[List[str], List[int]]:
        """Return (sample_ids, labels) of all currently labeled pool samples."""
        with self.get_session() as session:
            stmt = (
                select(SampleRecord.sample_id, SampleRecord.label)
                .where(
                    SampleRecord.experiment_id == self.experiment_id,
                    SampleRecord.split == "pool",
                    SampleRecord.status == "labeled",
                )
                .order_by(SampleRecord.sample_id)
            )
            rows = session.execute(stmt).all()
            return [r[0] for r in rows], [r[1] for r in rows]

    def get_split_samples(self, split_name: str) -> Tuple[List[str], List[int]]:
        """Return (sample_ids, labels) for a fixed split ('val' or 'test')."""
        with self.get_session() as session:
            stmt = (
                select(SampleRecord.sample_id, SampleRecord.true_label)
                .where(
                    SampleRecord.experiment_id == self.experiment_id,
                    SampleRecord.split == split_name,
                )
                .order_by(SampleRecord.sample_id)
            )
            rows = session.execute(stmt).all()
            return [r[0] for r in rows], [r[1] for r in rows]

    def mark_queued(
        self,
        sample_ids: List[str],
        batch_id: Optional[str] = None,
    ) -> None:
        """Mark samples as queued for labeling."""
        if not sample_ids:
            return
        now = _utcnow()
        with self.get_session() as session:
            values: Dict[str, Any] = {"status": "queued", "updated_at": now}
            if batch_id is not None:
                values["batch_id"] = batch_id
            session.execute(
                update(SampleRecord)
                .where(
                    SampleRecord.experiment_id == self.experiment_id,
                    SampleRecord.sample_id.in_(sample_ids),
                )
                .values(**values)
            )
            session.commit()

    def store_pre_annotations(
        self,
        pre_annotations: List[Dict[str, Any]],
    ) -> None:
        """Persist model predictions and Label Studio task mappings without confirming ground-truth labels."""
        if not pre_annotations:
            return
        now = _utcnow()
        with self.get_session() as session:
            for item in pre_annotations:
                values: Dict[str, Any] = {"updated_at": now}
                if "predicted_label" in item:
                    values["pre_annotation_label"] = item["predicted_label"]
                if "confidence" in item:
                    values["pre_annotation_confidence"] = item["confidence"]
                if "probabilities" in item and item["probabilities"] is not None:
                    values["pre_annotation_probs_json"] = json.dumps(item["probabilities"])
                if "task_id" in item and item["task_id"] is not None:
                    values["label_studio_task_id"] = int(item["task_id"])
                if "batch_id" in item and item["batch_id"] is not None:
                    values["batch_id"] = item["batch_id"]

                session.execute(
                    update(SampleRecord)
                    .where(
                        SampleRecord.experiment_id == self.experiment_id,
                        SampleRecord.sample_id == item["sample_id"],
                    )
                    .values(**values)
                )
            session.commit()

    def mark_labeled(
        self,
        sample_ids: List[str],
        labels: List[int],
        round_num: int,
        annotator: str = "oracle",
    ) -> None:
        """Mark samples as labeled and update their assigned labels."""
        now = _utcnow()
        with self.get_session() as session:
            for sid, lbl in zip(sample_ids, labels):
                session.execute(
                    update(SampleRecord)
                    .where(
                        SampleRecord.experiment_id == self.experiment_id,
                        SampleRecord.sample_id == sid,
                    )
                    .values(
                        status="labeled",
                        label=lbl,
                        round_added=round_num,
                        annotator=annotator,
                        updated_at=now,
                    )
                )
            session.commit()

    def record_human_annotation(
        self,
        sample_id: str,
        label_id: int,
        annotator: str,
        round_num: int,
        source_type: str = "human",
        batch_id: Optional[str] = None,
        webhook_event_id: Optional[str] = None,
        label_studio_annotation_id: Optional[int] = None,
        is_double_annotation: bool = False,
    ) -> Tuple[bool, Optional[str]]:
        """Persist a single annotation event transactionally with webhook idempotency.

        Returns:
            (inserted_or_updated_bool, reason_or_duplicate_note)
        """
        with self.get_session() as session:
            try:
                if webhook_event_id:
                    existing_event = (
                        session.query(AnnotationRecord)
                        .filter(AnnotationRecord.webhook_event_id == webhook_event_id)
                        .first()
                    )
                    if existing_event is not None:
                        return False, f"Duplicate webhook_event_id '{webhook_event_id}' ignored."

                sample = (
                    session.query(SampleRecord)
                    .filter(
                        SampleRecord.experiment_id == self.experiment_id,
                        SampleRecord.sample_id == sample_id,
                    )
                    .first()
                )
                if sample is None or sample.split != "pool":
                    raise KeyError(f"Pool sample '{sample_id}' not found.")

                existing_user_ann = (
                    session.query(AnnotationRecord)
                    .filter(
                        AnnotationRecord.experiment_id == self.experiment_id,
                        AnnotationRecord.sample_id == sample_id,
                        AnnotationRecord.annotator == annotator,
                        AnnotationRecord.round_num == round_num,
                    )
                    .first()
                )
                if existing_user_ann is not None:
                    # Update annotator's latest choice cleanly without inserting duplicate rows
                    existing_user_ann.label_id = label_id
                    existing_user_ann.source_type = source_type
                    if label_studio_annotation_id is not None:
                        existing_user_ann.label_studio_annotation_id = label_studio_annotation_id
                else:
                    ann_rec = AnnotationRecord(
                        experiment_id=self.experiment_id,
                        sample_id=sample_id,
                        batch_id=batch_id or sample.batch_id,
                        round_num=round_num,
                        annotator=annotator,
                        label_id=label_id,
                        source_type=source_type,
                        is_double_annotation=is_double_annotation,
                        webhook_event_id=webhook_event_id,
                        label_studio_annotation_id=label_studio_annotation_id,
                    )
                    session.add(ann_rec)

                # Transition sample status queued -> labeled (or update label)
                prior_annotations = (
                    session.query(AnnotationRecord)
                    .filter(
                        AnnotationRecord.experiment_id == self.experiment_id,
                        AnnotationRecord.sample_id == sample_id,
                    )
                    .all()
                )
                distinct_labels = {a.label_id for a in prior_annotations}
                distinct_labels.add(label_id)
                if len(distinct_labels) > 1:
                    sample.needs_review = True

                sample.status = "labeled"
                sample.label = label_id
                sample.round_added = round_num
                sample.annotator = annotator
                sample.updated_at = _utcnow()

                session.commit()
                return True, None
            except IntegrityError:
                session.rollback()
                return False, "Duplicate webhook/annotation event rejected by unique constraint."
            except Exception:
                session.rollback()
                raise

    def get_annotation_records(
        self,
        round_num: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Return all individual annotation events for QA / Cohen's Kappa evaluation."""
        with self.get_session() as session:
            q = session.query(AnnotationRecord).filter(
                AnnotationRecord.experiment_id == self.experiment_id
            )
            if round_num is not None:
                q = q.filter(AnnotationRecord.round_num == round_num)
            rows = q.order_by(AnnotationRecord.created_at.asc()).all()
            return [
                {
                    "id": r.id,
                    "sample_id": r.sample_id,
                    "batch_id": r.batch_id,
                    "round_num": r.round_num,
                    "annotator": r.annotator,
                    "label_id": r.label_id,
                    "source_type": r.source_type,
                    "is_double_annotation": bool(r.is_double_annotation),
                    "webhook_event_id": r.webhook_event_id,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                }
                for r in rows
            ]

    def create_or_update_batch(
        self,
        batch_id: str,
        round_num: int,
        strategy: str,
        total_tasks: int,
        threshold_count: int,
        double_annotation_rate: float = 0.10,
    ) -> None:
        with self.get_session() as session:
            batch = session.get(AnnotationBatchRecord, batch_id)
            if batch is None:
                batch = AnnotationBatchRecord(
                    batch_id=batch_id,
                    experiment_id=self.experiment_id,
                    round_num=round_num,
                    strategy=strategy,
                    total_tasks=total_tasks,
                    completed_tasks=0,
                    threshold_count=threshold_count,
                    double_annotation_rate=double_annotation_rate,
                    status="pending",
                )
                session.add(batch)
            else:
                batch.total_tasks = total_tasks
                batch.threshold_count = threshold_count
            session.commit()

    def refresh_batch_progress(self, batch_id: str) -> Optional[Dict[str, Any]]:
        with self.get_session() as session:
            batch = session.get(AnnotationBatchRecord, batch_id)
            if batch is None:
                return None
            completed = (
                session.query(SampleRecord)
                .filter(
                    SampleRecord.experiment_id == self.experiment_id,
                    SampleRecord.batch_id == batch_id,
                    SampleRecord.status == "labeled",
                )
                .count()
            )
            batch.completed_tasks = completed
            if completed >= batch.threshold_count and batch.status == "pending":
                batch.status = "ready_to_retrain"
                batch.completed_at = _utcnow()
            session.commit()
            return {
                "batch_id": batch.batch_id,
                "round_num": batch.round_num,
                "total_tasks": batch.total_tasks,
                "completed_tasks": batch.completed_tasks,
                "threshold_count": batch.threshold_count,
                "status": batch.status,
            }

    def save_quality_metric(self, metrics: Dict[str, Any], round_num: int = 0) -> None:
        with self.get_session() as session:
            rec = QualityMetricRecord(
                experiment_id=self.experiment_id,
                round_num=round_num,
                n_overlapping=int(metrics.get("n_overlapping", metrics.get("overlapping_samples", 0))),
                cohens_kappa=metrics.get("cohens_kappa"),
                raw_agreement=metrics.get("raw_agreement", metrics.get("raw_agreement_rate")),
                disagreement_count=int(metrics.get("disagreement_count", 0)),
                flagged_sample_ids_json=json.dumps(
                    metrics.get("flagged_sample_ids", [d.get("sample_id") for d in metrics.get("disagreements", [])])
                ),
                per_class_json=json.dumps(metrics.get("per_class_agreement", metrics.get("class_wise_agreement", []))),
                annotator_stats_json=json.dumps(metrics.get("annotator_stats", [])),
            )
            session.add(rec)
            session.commit()

    def save_model_version(
        self,
        round_num: int,
        model_type: str,
        checkpoint_path: str,
        temperature: Optional[float] = None,
        val_ece: Optional[float] = None,
        test_acc: Optional[float] = None,
        test_f1: Optional[float] = None,
        test_ece: Optional[float] = None,
        test_nll: Optional[float] = None,
        train_time_sec: Optional[float] = None,
        mlflow_run_id: Optional[str] = None,
    ) -> None:
        with self.get_session() as session:
            rec = ModelVersionRecord(
                experiment_id=self.experiment_id,
                round_num=round_num,
                model_type=model_type,
                checkpoint_path=checkpoint_path,
                temperature=temperature,
                val_ece=val_ece,
                test_acc=test_acc,
                test_f1=test_f1,
                test_ece=test_ece,
                test_nll=test_nll,
                train_time_sec=train_time_sec,
                mlflow_run_id=mlflow_run_id,
            )
            session.add(rec)
            session.commit()

    def get_latest_model_version(self) -> Optional[Dict[str, Any]]:
        with self.get_session() as session:
            row = (
                session.query(ModelVersionRecord)
                .filter(ModelVersionRecord.experiment_id == self.experiment_id)
                .order_by(ModelVersionRecord.round_num.desc(), ModelVersionRecord.id.desc())
                .first()
            )
            if row is None:
                return None
            return {
                "round_num": row.round_num,
                "model_type": row.model_type,
                "checkpoint_path": row.checkpoint_path,
                "temperature": row.temperature,
                "test_acc": row.test_acc,
                "test_f1": row.test_f1,
                "test_ece": row.test_ece,
                "test_nll": row.test_nll,
                "mlflow_run_id": row.mlflow_run_id,
                "created_at": row.created_at.isoformat() if row.created_at else None,
            }

    def record_continuous_feedback(
        self,
        sample_id: str,
        human_label_id: int,
        human_label_name: str,
        annotator: str,
        original_pred_id: Optional[int] = None,
        original_pred_name: Optional[str] = None,
        original_confidence: Optional[float] = None,
        model_version_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Record human verification/correction for continuous learning."""
        is_corr = bool(original_pred_id is not None and original_pred_id != human_label_id)
        now = _utcnow()
        with self.get_session() as session:
            rec = ContinuousFeedbackRecord(
                experiment_id=self.experiment_id,
                sample_id=sample_id,
                original_pred_id=original_pred_id,
                original_pred_name=original_pred_name,
                original_confidence=original_confidence,
                human_label_id=human_label_id,
                human_label_name=human_label_name,
                is_correction=is_corr,
                annotator=annotator,
                model_version_id=model_version_id,
                used_in_training=False,
                created_at=now,
            )
            session.add(rec)
            session.commit()
            return {
                "id": rec.id,
                "sample_id": rec.sample_id,
                "is_correction": rec.is_correction,
                "human_label_id": rec.human_label_id,
                "human_label_name": rec.human_label_name,
            }

    def mark_feedback_used_in_training(self, feedback_ids: List[int], job_id: str) -> None:
        if not feedback_ids:
            return
        with self.get_session() as session:
            session.execute(
                update(ContinuousFeedbackRecord)
                .where(ContinuousFeedbackRecord.id.in_(feedback_ids))
                .values(used_in_training=True, retrain_job_id=job_id)
            )
            session.commit()

    def get_continuous_feedback_stats(self) -> Dict[str, int]:
        with self.get_session() as session:
            total_verified = (
                session.query(ContinuousFeedbackRecord)
                .filter(ContinuousFeedbackRecord.experiment_id == self.experiment_id)
                .count()
            )
            corrections = (
                session.query(ContinuousFeedbackRecord)
                .filter(
                    ContinuousFeedbackRecord.experiment_id == self.experiment_id,
                    ContinuousFeedbackRecord.is_correction == True,  # noqa: E712
                )
                .count()
            )
            confirmed = total_verified - corrections
            pending_retrain = (
                session.query(ContinuousFeedbackRecord)
                .filter(
                    ContinuousFeedbackRecord.experiment_id == self.experiment_id,
                    ContinuousFeedbackRecord.used_in_training == False,  # noqa: E712
                )
                .count()
            )
            trained = (
                session.query(ContinuousFeedbackRecord)
                .filter(
                    ContinuousFeedbackRecord.experiment_id == self.experiment_id,
                    ContinuousFeedbackRecord.used_in_training == True,  # noqa: E712
                )
                .count()
            )
            return {
                "total_verified": total_verified,
                "corrections": corrections,
                "confirmed": confirmed,
                "pending_retrain": pending_retrain,
                "used_in_training": trained,
            }

    def get_pending_continuous_feedback(self) -> List[Dict[str, Any]]:
        with self.get_session() as session:
            rows = (
                session.query(ContinuousFeedbackRecord)
                .filter(
                    ContinuousFeedbackRecord.experiment_id == self.experiment_id,
                    ContinuousFeedbackRecord.used_in_training == False,  # noqa: E712
                )
                .order_by(ContinuousFeedbackRecord.created_at.asc())
                .all()
            )
            return [
                {
                    "id": r.id,
                    "sample_id": r.sample_id,
                    "human_label_id": r.human_label_id,
                    "human_label_name": r.human_label_name,
                    "is_correction": r.is_correction,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                }
                for r in rows
            ]

    def record_retrain_event(
        self,
        event_id: str,
        trigger_type: str,
        trigger_sample_id: Optional[str] = None,
        samples_incorporated: int = 0,
        total_training_samples: int = 0,
        prev_model_version: Optional[str] = None,
        candidate_model_version: Optional[str] = None,
        prev_accuracy: Optional[float] = None,
        prev_macro_f1: Optional[float] = None,
        prev_ece: Optional[float] = None,
        status: str = "pending",
        decision_reason: Optional[str] = None,
    ) -> None:
        now = _utcnow()
        with self.get_session() as session:
            rec = ContinuousRetrainEventRecord(
                event_id=event_id,
                experiment_id=self.experiment_id,
                trigger_type=trigger_type,
                trigger_sample_id=trigger_sample_id,
                samples_incorporated=samples_incorporated,
                total_training_samples=total_training_samples,
                prev_model_version=prev_model_version,
                candidate_model_version=candidate_model_version,
                prev_accuracy=prev_accuracy,
                prev_macro_f1=prev_macro_f1,
                prev_ece=prev_ece,
                status=status,
                decision_reason=decision_reason,
                created_at=now,
            )
            session.add(rec)
            session.commit()

    def update_retrain_event(
        self,
        event_id: str,
        candidate_accuracy: Optional[float] = None,
        candidate_macro_f1: Optional[float] = None,
        candidate_ece: Optional[float] = None,
        status: Optional[str] = None,
        decision_reason: Optional[str] = None,
    ) -> None:
        now = _utcnow()
        with self.get_session() as session:
            rec = (
                session.query(ContinuousRetrainEventRecord)
                .filter(ContinuousRetrainEventRecord.event_id == event_id)
                .first()
            )
            if rec:
                if candidate_accuracy is not None:
                    rec.candidate_accuracy = candidate_accuracy
                if candidate_macro_f1 is not None:
                    rec.candidate_macro_f1 = candidate_macro_f1
                if candidate_ece is not None:
                    rec.candidate_ece = candidate_ece
                if status is not None:
                    rec.status = status
                    if status in ("promoted", "rejected", "failed"):
                        rec.completed_at = now
                if decision_reason is not None:
                    rec.decision_reason = decision_reason
                session.commit()

    def list_retrain_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self.get_session() as session:
            rows = (
                session.query(ContinuousRetrainEventRecord)
                .filter(ContinuousRetrainEventRecord.experiment_id == self.experiment_id)
                .order_by(ContinuousRetrainEventRecord.created_at.desc())
                .limit(limit)
                .all()
            )
            return [
                {
                    "event_id": r.event_id,
                    "trigger_type": r.trigger_type,
                    "trigger_sample_id": r.trigger_sample_id,
                    "samples_incorporated": r.samples_incorporated,
                    "total_training_samples": r.total_training_samples,
                    "prev_model_version": r.prev_model_version,
                    "candidate_model_version": r.candidate_model_version,
                    "prev_accuracy": r.prev_accuracy,
                    "prev_macro_f1": r.prev_macro_f1,
                    "prev_ece": r.prev_ece,
                    "candidate_accuracy": r.candidate_accuracy,
                    "candidate_macro_f1": r.candidate_macro_f1,
                    "candidate_ece": r.candidate_ece,
                    "status": r.status,
                    "decision_reason": r.decision_reason,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                    "completed_at": r.completed_at.isoformat() if r.completed_at else None,
                }
                for r in rows
            ]

    def record_job(
        self,
        job_id: str,
        job_type: str,
        status: str,
        progress_pct: float = 0.0,
        current_stage: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> None:
        now = _utcnow()
        with self.get_session() as session:
            job = session.get(ExperimentJobRecord, job_id)
            if job is None:
                job = ExperimentJobRecord(
                    job_id=job_id,
                    experiment_id=self.experiment_id,
                    job_type=job_type,
                    status=status,
                    progress_pct=progress_pct,
                    current_stage=current_stage,
                    error_message=error_message,
                    started_at=now if status == "running" else None,
                    completed_at=now if status in ("completed", "failed", "cancelled") else None,
                )
                session.add(job)
            else:
                job.status = status
                job.progress_pct = progress_pct
                if current_stage is not None:
                    job.current_stage = current_stage
                if error_message is not None:
                    job.error_message = error_message
                if status == "running" and job.started_at is None:
                    job.started_at = now
                if status in ("completed", "failed", "cancelled"):
                    job.completed_at = now
            session.commit()

    def list_jobs(self) -> List[Dict[str, Any]]:
        with self.get_session() as session:
            rows = (
                session.query(ExperimentJobRecord)
                .filter(ExperimentJobRecord.experiment_id == self.experiment_id)
                .order_by(ExperimentJobRecord.created_at.desc())
                .all()
            )
            return [
                {
                    "job_id": j.job_id,
                    "experiment_id": j.experiment_id,
                    "job_type": j.job_type,
                    "status": j.status,
                    "progress_pct": float(j.progress_pct or 0.0),
                    "current_stage": j.current_stage,
                    "error_message": j.error_message,
                    "created_at": j.created_at.isoformat() if j.created_at else None,
                    "updated_at": (j.completed_at or j.started_at or j.created_at).isoformat()
                    if (j.completed_at or j.started_at or j.created_at)
                    else None,
                }
                for j in rows
            ]

    def save_experiment_meta(
        self,
        name: str,
        dataset: str,
        model_type: str,
        strategy: str,
        mode: str = "simulated_oracle",
        status: str = "created",
        config: Optional[Dict[str, Any]] = None,
        current_round: int = 0,
        max_rounds: int = 3,
        current_stage: str = "Initialized",
        stage_index: int = 0,
        stop_reason: Optional[str] = None,
        mlflow_run_id: Optional[str] = None,
        label_studio_project_id: Optional[int] = None,
        error_message: Optional[str] = None,
    ) -> None:
        """Upsert experiment metadata row in the relational database."""
        now = _utcnow()
        with self.get_session() as session:
            rec = session.get(ExperimentRecord, self.experiment_id)
            if rec is None:
                rec = ExperimentRecord(
                    id=self.experiment_id,
                    name=name,
                    dataset=dataset,
                    model_type=model_type,
                    strategy=strategy,
                    mode=mode,
                    status=status,
                    current_round=current_round,
                    max_rounds=max_rounds,
                    current_stage=current_stage,
                    stage_index=stage_index,
                    config_json=json.dumps(config) if config is not None else None,
                    stop_reason=stop_reason,
                    mlflow_run_id=mlflow_run_id,
                    label_studio_project_id=label_studio_project_id,
                    error_message=error_message,
                    created_at=now,
                    updated_at=now,
                )
                session.add(rec)
            else:
                rec.name = name
                rec.dataset = dataset
                rec.model_type = model_type
                rec.strategy = strategy
                rec.mode = mode
                rec.status = status
                rec.current_round = current_round
                rec.max_rounds = max_rounds
                rec.current_stage = current_stage
                rec.stage_index = stage_index
                if config is not None:
                    rec.config_json = json.dumps(config)
                if stop_reason is not None:
                    rec.stop_reason = stop_reason
                if mlflow_run_id is not None:
                    rec.mlflow_run_id = mlflow_run_id
                if label_studio_project_id is not None:
                    rec.label_studio_project_id = label_studio_project_id
                rec.error_message = error_message
                rec.updated_at = now
            session.commit()

    def record_round_metrics(
        self,
        round_num: int,
        strategy: str,
        n_labeled: int,
        n_pool: int,
        test_acc: float,
        test_f1: float,
        ece: Optional[float] = None,
        temperature: Optional[float] = None,
        nll: Optional[float] = None,
        train_time_sec: Optional[float] = None,
    ) -> None:
        """Save evaluation metrics for a completed AL round."""
        with self.get_session() as session:
            existing = (
                session.query(RoundRecord)
                .filter(
                    RoundRecord.experiment_id == self.experiment_id,
                    RoundRecord.round_num == round_num,
                )
                .first()
            )
            if existing:
                existing.strategy = strategy
                existing.n_labeled = n_labeled
                existing.n_pool = n_pool
                existing.test_acc = test_acc
                existing.test_f1 = test_f1
                existing.ece = ece
                existing.temperature = temperature
                existing.nll = nll
                existing.train_time_sec = train_time_sec
                existing.timestamp = _utcnow()
            else:
                record = RoundRecord(
                    experiment_id=self.experiment_id,
                    round_num=round_num,
                    strategy=strategy,
                    n_labeled=n_labeled,
                    n_pool=n_pool,
                    test_acc=test_acc,
                    test_f1=test_f1,
                    ece=ece,
                    temperature=temperature,
                    nll=nll,
                    train_time_sec=train_time_sec,
                )
                session.add(record)
            session.commit()

    def get_latest_round(self) -> int:
        """Get highest completed round number, or -1 if none."""
        with self.get_session() as session:
            stmt = (
                select(RoundRecord.round_num)
                .where(RoundRecord.experiment_id == self.experiment_id)
                .order_by(RoundRecord.round_num.desc())
                .limit(1)
            )
            row = session.execute(stmt).first()
            return row[0] if row else -1

    def get_all_rounds_history(self) -> List[Dict[str, Any]]:
        """Return history of all recorded rounds as a list of dictionaries."""
        with self.get_session() as session:
            stmt = (
                select(RoundRecord)
                .where(RoundRecord.experiment_id == self.experiment_id)
                .order_by(RoundRecord.round_num.asc())
            )
            rows = session.execute(stmt).scalars().all()
            return [
                {
                    "round_num": r.round_num,
                    "strategy": r.strategy,
                    "n_labeled": r.n_labeled,
                    "n_pool": r.n_pool,
                    "test_acc": r.test_acc,
                    "test_f1": r.test_f1,
                    "ece": r.ece,
                    "temperature": r.temperature,
                    "nll": r.nll,
                    "train_time_sec": r.train_time_sec,
                    "timestamp": r.timestamp.isoformat() if r.timestamp else None,
                }
                for r in rows
            ]

    def reset_database(self) -> None:
        """Drop all tables and recreate clean schema."""
        Base.metadata.drop_all(self.engine)
        Base.metadata.create_all(self.engine)

    def close(self) -> None:
        """Dispose of the database engine connection pool."""
        self.engine.dispose()
