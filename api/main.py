from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from typing import Optional
from api.schemas import AtmosphericReport
from core.pipeline import run_atmospheric_analysis

app = FastAPI(
    title="AtmosVision API",
    description="Extracts atmospheric physics, cloud instability, radiation, and wind dynamics from outdoor images.",
    version="0.1.0",
)


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.post("/analyze", response_model=AtmosphericReport)
async def analyze_image(
    file: UploadFile = File(...),
    override_lat: Optional[float] = Form(None),
    override_lon: Optional[float] = Form(None),
    override_timestamp_iso: Optional[str] = Form(None),
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    image_bytes = await file.read()
    report = await run_atmospheric_analysis(
        image_bytes=image_bytes,
        override_lat=override_lat,
        override_lon=override_lon,
        override_timestamp_iso=override_timestamp_iso,
    )
    return report
