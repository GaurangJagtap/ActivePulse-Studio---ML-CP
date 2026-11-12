"""Health and system status endpoint."""

from fastapi import APIRouter
import torch

from backend.api.config import API_VERSION, DEFAULT_DEVICE, LABEL_STUDIO_URL
from backend.api.schemas.common import HealthResponse
from src.strategies.registry import list_registered_strategies

router = APIRouter(tags=["System"])


@router.get("/health", response_model=HealthResponse, summary="Check backend & ML engine health")
def get_health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        service="ActivePulse Studio API",
        version=API_VERSION,
        pytorch_version=torch.__version__,
        device=DEFAULT_DEVICE,
        registered_strategies=len(list_registered_strategies()),
        label_studio_configured=bool(LABEL_STUDIO_URL),
    )

