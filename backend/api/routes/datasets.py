"""Dataset inspection and sampling preview endpoints."""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query

from backend.api.schemas.common import (
    DatasetSamplesResponse,
    DatasetStatsResponse,
    DatasetSummary,
)
from backend.api.services.catalog_service import get_datasets_catalog
from backend.api.services.dataset_service import get_dataset_samples, get_dataset_stats
from backend.api.services.experiment_service import get_experiment_db

router = APIRouter(prefix="/datasets", tags=["Datasets"])


@router.get("", response_model=List[DatasetSummary], summary="List supported datasets")
def list_datasets() -> List[DatasetSummary]:
    return get_datasets_catalog()


@router.get("/{dataset_id}/stats", response_model=DatasetStatsResponse, summary="Get dataset split & class statistics")
def dataset_stats(
    dataset_id: str,
    max_samples: int = Query(default=450, ge=50, le=5000),
    use_synthetic: bool = Query(default=True),
    seed: int = Query(default=42),
    experiment_id: Optional[str] = Query(default=None),
) -> DatasetStatsResponse:
    try:
        exp_db = get_experiment_db(experiment_id) if experiment_id else None
        return get_dataset_stats(
            dataset_id=dataset_id,
            max_samples=max_samples,
            use_synthetic=use_synthetic,
            seed=seed,
            experiment_db=exp_db,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/{dataset_id}/samples", response_model=DatasetSamplesResponse, summary="Browse dataset samples")
def dataset_samples(
    dataset_id: str,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=24, ge=1, le=100),
    max_samples: int = Query(default=450, ge=50, le=5000),
    use_synthetic: bool = Query(default=True),
    seed: int = Query(default=42),
    split: Optional[str] = Query(default=None),
    status: Optional[str] = Query(default=None),
    experiment_id: Optional[str] = Query(default=None),
) -> DatasetSamplesResponse:
    try:
        exp_db = get_experiment_db(experiment_id) if experiment_id else None
        return get_dataset_samples(
            dataset_id=dataset_id,
            offset=offset,
            limit=limit,
            max_samples=max_samples,
            use_synthetic=use_synthetic,
            seed=seed,
            split_filter=split,
            status_filter=status,
            experiment_db=exp_db,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

