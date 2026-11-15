"""Dataset inspection and sampling preview endpoints."""

from typing import List, Optional
from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile

from backend.api.schemas.common import (
    DatasetSamplesResponse,
    DatasetStatsResponse,
    DatasetSummary,
)
from backend.api.services.catalog_service import get_datasets_catalog
from backend.api.services.dataset_service import get_dataset_samples, get_dataset_stats
from backend.api.services.experiment_service import get_experiment_db
from backend.api.services.upload_service import (
    process_csv_text_upload,
    process_image_zip_upload,
)

router = APIRouter(prefix="/datasets", tags=["Datasets"])


@router.post("/upload", response_model=DatasetSummary, summary="Upload a custom dataset (ZIP images or CSV text)")
async def upload_dataset(
    file: UploadFile = File(...),
    name: str = Form(...),
    modality: str = Form(default="image"),
    text_column: str = Form(default="text"),
    label_column: Optional[str] = Form(default="label"),
) -> DatasetSummary:
    try:
        content = await file.read()
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        filename = (file.filename or "").lower()
        if modality == "image" or filename.endswith(".zip"):
            if not filename.endswith(".zip"):
                raise HTTPException(status_code=400, detail="Custom image dataset must be uploaded as a .zip file.")
            meta = process_image_zip_upload(zip_bytes=content, dataset_name=name)
        elif modality == "text" or filename.endswith(".csv"):
            if not filename.endswith(".csv"):
                raise HTTPException(status_code=400, detail="Custom text dataset must be uploaded as a .csv file.")
            meta = process_csv_text_upload(
                csv_bytes=content,
                dataset_name=name,
                text_column=text_column,
                label_column=label_column,
            )
        else:
            raise HTTPException(status_code=400, detail="Unsupported modality or file extension. Use .zip (images) or .csv (text).")

        supported_models = ["cifar_cnn", "resnet18"] if meta["modality"] == "image" else ["text_mlp"]
        return DatasetSummary(
            id=meta["id"],
            name=f"{meta['name']} (Custom Upload)",
            modality=meta["modality"],
            num_classes=meta["num_classes"],
            class_names=meta["class_names"],
            default_model=supported_models[0],
            supported_models=supported_models,
            supports_real_download=False,
            supports_raw_text=(meta["modality"] == "text"),
            description=meta["description"],
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Upload processing failed: {exc}") from exc



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

