"""Database persistence layer for Active Learning samples and rounds state."""

from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    String,
    create_engine,
    select,
    update,
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session

Base = declarative_base()


class SampleRecord(Base):
    """SQLAlchemy model for dataset sample tracking across active learning rounds."""

    __tablename__ = "samples"

    sample_id = Column(String(64), primary_key=True, index=True)
    split = Column(String(32), nullable=False, index=True)  # 'pool', 'val', 'test'
    status = Column(String(32), nullable=False, default="unlabeled", index=True)  # 'unlabeled', 'queued', 'labeled'
    label = Column(Integer, nullable=True)  # Assigned label
    true_label = Column(Integer, nullable=False)  # Ground-truth label (for oracle)
    round_added = Column(Integer, nullable=True)  # AL round in which sample was labeled
    annotator = Column(String(64), nullable=True)  # E.g., 'simulated_oracle', 'human_1'
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class RoundRecord(Base):
    """SQLAlchemy model for recording round metrics."""

    __tablename__ = "rounds"

    round_num = Column(Integer, primary_key=True)
    strategy = Column(String(64), nullable=False)
    n_labeled = Column(Integer, nullable=False)
    n_pool = Column(Integer, nullable=False)
    test_acc = Column(Float, nullable=False)
    test_f1 = Column(Float, nullable=False)
    ece = Column(Float, nullable=True)
    temperature = Column(Float, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)


class DatabaseManager:
    """Manages SQLite / PostgreSQL connection and queries for AL pool and round state."""

    def __init__(self, db_url: str = "sqlite:///al_platform.db"):
        self.db_url = db_url
        self.engine = create_engine(
            db_url,
            connect_args={"check_same_thread": False} if db_url.startswith("sqlite") else {},
            echo=False,
        )
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine, autoflush=False, autocommit=False)

    def get_session(self) -> Session:
        return self.SessionLocal()

    def initialize_samples(
        self,
        sample_ids: List[str],
        splits: List[str],
        true_labels: List[int],
    ) -> None:
        """Populate database with sample metadata if not already present."""
        with self.get_session() as session:
            existing_count = session.query(SampleRecord).count()
            if existing_count > 0:
                return  # Database already initialized

            records = [
                SampleRecord(
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

    def get_pool_sample_ids(self) -> List[str]:
        """Return all sample_ids currently in the unlabeled pool."""
        with self.get_session() as session:
            stmt = (
                select(SampleRecord.sample_id)
                .where(SampleRecord.split == "pool", SampleRecord.status == "unlabeled")
                .order_by(SampleRecord.sample_id)
            )
            return [row[0] for row in session.execute(stmt).all()]

    def get_labeled_samples(self) -> Tuple[List[str], List[int]]:
        """Return (sample_ids, labels) of all currently labeled pool samples."""
        with self.get_session() as session:
            stmt = (
                select(SampleRecord.sample_id, SampleRecord.label)
                .where(SampleRecord.split == "pool", SampleRecord.status == "labeled")
                .order_by(SampleRecord.sample_id)
            )
            rows = session.execute(stmt).all()
            return [r[0] for r in rows], [r[1] for r in rows]

    def get_split_samples(self, split_name: str) -> Tuple[List[str], List[int]]:
        """Return (sample_ids, labels) for a fixed split ('val' or 'test')."""
        with self.get_session() as session:
            stmt = (
                select(SampleRecord.sample_id, SampleRecord.true_label)
                .where(SampleRecord.split == split_name)
                .order_by(SampleRecord.sample_id)
            )
            rows = session.execute(stmt).all()
            return [r[0] for r in rows], [r[1] for r in rows]

    def mark_queued(self, sample_ids: List[str]) -> None:
        """Mark samples as queued for labeling."""
        with self.get_session() as session:
            session.execute(
                update(SampleRecord)
                .where(SampleRecord.sample_id.in_(sample_ids))
                .values(status="queued", updated_at=datetime.utcnow())
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
        with self.get_session() as session:
            for sid, lbl in zip(sample_ids, labels):
                session.execute(
                    update(SampleRecord)
                    .where(SampleRecord.sample_id == sid)
                    .values(
                        status="labeled",
                        label=lbl,
                        round_added=round_num,
                        annotator=annotator,
                        updated_at=datetime.utcnow(),
                    )
                )
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
    ) -> None:
        """Save evaluation metrics for a completed AL round."""
        with self.get_session() as session:
            # Overwrite if round already exists (e.g., resumed or re-run)
            existing = session.get(RoundRecord, round_num)
            if existing:
                existing.strategy = strategy
                existing.n_labeled = n_labeled
                existing.n_pool = n_pool
                existing.test_acc = test_acc
                existing.test_f1 = test_f1
                existing.ece = ece
                existing.temperature = temperature
                existing.timestamp = datetime.utcnow()
            else:
                record = RoundRecord(
                    round_num=round_num,
                    strategy=strategy,
                    n_labeled=n_labeled,
                    n_pool=n_pool,
                    test_acc=test_acc,
                    test_f1=test_f1,
                    ece=ece,
                    temperature=temperature,
                )
                session.add(record)
            session.commit()

    def get_latest_round(self) -> int:
        """Get highest completed round number, or -1 if none."""
        with self.get_session() as session:
            stmt = select(RoundRecord.round_num).order_by(RoundRecord.round_num.desc()).limit(1)
            row = session.execute(stmt).first()
            return row[0] if row else -1

    def get_all_rounds_history(self) -> List[Dict[str, Any]]:
        """Return history of all recorded rounds as a list of dictionaries."""
        with self.get_session() as session:
            stmt = select(RoundRecord).order_by(RoundRecord.round_num.asc())
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
