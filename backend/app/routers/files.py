from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import FileResponse

from app.storage import content_type_for_file, safe_resolve

router = APIRouter(tags=["files"])


@router.get("/files/{file_path:path}")
async def serve_file(file_path: str):
    resolved = safe_resolve(file_path)
    if resolved is None or not resolved.exists() or not resolved.is_file():
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(
        path=str(resolved),
        media_type=content_type_for_file(resolved),
        filename=resolved.name,
    )
