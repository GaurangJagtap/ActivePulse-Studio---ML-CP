"""Initial ActivePulse Studio relational schema (experiments, samples, rounds, batches, annotations, quality, jobs, model versions).

Revision ID: 20261112_0001
Revises:
Create Date: 2026-11-12
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

from src.data.db import Base

revision = "20261112_0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    Base.metadata.create_all(bind=bind, checkfirst=True)


def downgrade() -> None:
    for table_name in (
        "model_versions",
        "experiment_jobs",
        "quality_metrics",
        "annotations",
        "annotation_batches",
        "rounds",
        "samples",
        "experiments",
    ):
        op.drop_table(table_name)

