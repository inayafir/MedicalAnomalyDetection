from fastapi import APIRouter

from app.ml_interface import is_model_loaded

router = APIRouter()


@router.get("/health")
async def health_check():
    return {"status": "ok", "model_loaded": is_model_loaded()}
