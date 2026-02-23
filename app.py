#!/usr/bin/env python3
"""
Edit Banana — FastAPI Backend (UI + API + GPU Manager)
Image/PDF → Editable DrawIO XML/PPTX via SAM3 segmentation
"""
import asyncio
import gc
import json
import os
import shutil
import sys
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

VERSION = "1.0.0"
ALLOWED_EXTS = {".png", ".jpg", ".jpeg", ".pdf", ".bmp", ".tiff", ".webp"}

# ─── GPU Manager ───────────────────────────────────────────
class GPUManager:
    def __init__(self):
        self.pipeline = None
        self.sam3_loaded = False
        self.last_used = 0.0
        self.lock = asyncio.Lock()
        self.idle_timeout = int(os.getenv("GPU_IDLE_TIMEOUT", "600"))

    async def get_pipeline(self):
        async with self.lock:
            if self.pipeline:
                self.last_used = time.time()
                return self.pipeline
            from main import Pipeline, load_config
            config = load_config()
            self.pipeline = Pipeline(config)
            self.sam3_loaded = True
            self.last_used = time.time()
            return self.pipeline

    async def offload(self):
        async with self.lock:
            if self.pipeline:
                del self.pipeline
                self.pipeline = None
                self.sam3_loaded = False
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return True
        return False

    def gpu_info(self):
        if not torch.cuda.is_available():
            return {"available": False}
        idx = torch.cuda.current_device()
        total = torch.cuda.get_device_properties(idx).total_memory / 1e9
        used = torch.cuda.memory_allocated(idx) / 1e9
        return {
            "available": True,
            "device": torch.cuda.get_device_name(idx),
            "memory_total_gb": round(total, 2),
            "memory_used_gb": round(used, 2),
            "memory_free_gb": round(total - used, 2),
        }

    async def auto_offload_loop(self):
        while True:
            await asyncio.sleep(60)
            if self.pipeline and (time.time() - self.last_used) > self.idle_timeout:
                await self.offload()

gpu_mgr = GPUManager()

# ─── App ───────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(gpu_mgr.auto_offload_loop())
    yield
    task.cancel()

app = FastAPI(
    title="Edit Banana",
    description="Image/PDF → Editable DrawIO XML via SAM3 segmentation",
    version=VERSION,
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# ─── Error handler ─────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"error": str(exc)})

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})

# ─── UI ────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def ui():
    html_path = os.path.join(PROJECT_ROOT, "ui.html")
    if os.path.exists(html_path):
        return HTMLResponse(open(html_path, encoding="utf-8").read())
    return HTMLResponse("<h1>Edit Banana</h1><p>UI not found</p>")

# ─── Health & Status ───────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "version": VERSION,
        "model_loaded": gpu_mgr.sam3_loaded,
        "gpu": gpu_mgr.gpu_info(),
    }

@app.get("/api/status")
async def status():
    return {
        "service": "Edit Banana",
        "version": VERSION,
        "model_loaded": gpu_mgr.sam3_loaded,
        "gpu": gpu_mgr.gpu_info(),
        "idle_timeout": gpu_mgr.idle_timeout,
        "supported_formats": list(ALLOWED_EXTS),
    }

# ─── Config ────────────────────────────────────────────────
@app.get("/api/config")
async def get_config():
    cfg_path = os.path.join(PROJECT_ROOT, "config", "config.yaml")
    if not os.path.exists(cfg_path):
        return {"error": "config not found"}
    import yaml
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # Strip secrets
    if "multimodal" in cfg:
        for k in ("api_key",):
            if k in cfg["multimodal"]:
                cfg["multimodal"][k] = "***"
    return cfg

# ─── Main Convert API ─────────────────────────────────────
@app.post("/api/convert")
async def convert(
    file: UploadFile = File(...),
    with_text: bool = Form(True),
    with_refinement: bool = Form(False),
    score_threshold: Optional[float] = Form(None),
    epsilon_factor: Optional[float] = Form(None),
    min_area: Optional[int] = Form(None),
):
    t0 = time.time()
    name = file.filename or "upload"
    ext = Path(name).suffix.lower()
    if ext not in ALLOWED_EXTS:
        raise HTTPException(400, f"Unsupported format '{ext}'. Use: {ALLOWED_EXTS}")

    # Save upload
    tmp_dir = tempfile.mkdtemp(prefix="eb_")
    tmp_path = os.path.join(tmp_dir, name)
    with open(tmp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    t_load = time.time()
    try:
        pipeline = await gpu_mgr.get_pipeline()

        # Override config params if provided
        if score_threshold is not None:
            pipeline.config.setdefault("sam3", {})["score_threshold"] = score_threshold
        if epsilon_factor is not None:
            pipeline.config.setdefault("sam3", {})["epsilon_factor"] = epsilon_factor
        if min_area is not None:
            pipeline.config.setdefault("sam3", {})["min_area"] = min_area

        output_dir = os.path.join(tmp_dir, "output")
        os.makedirs(output_dir, exist_ok=True)

        result_path = pipeline.process_image(
            tmp_path, output_dir=output_dir,
            with_refinement=with_refinement, with_text=with_text,
        )
        t_proc = time.time()

        if not result_path or not os.path.exists(result_path):
            raise HTTPException(500, "Conversion failed — no output generated")

        # Read result
        with open(result_path, encoding="utf-8") as f:
            xml_content = f.read()

        # Collect visualization if exists
        stem = Path(tmp_path).stem
        vis_path = os.path.join(output_dir, stem, "sam3_extraction.png")

        resp = {
            "success": True,
            "filename": name,
            "xml_content": xml_content,
            "elements_count": xml_content.count("<mxCell"),
            "processing_time": round(t_proc - t_load, 2),
        }
        headers = {
            "X-Time-Load": f"{t_load - t0:.3f}",
            "X-Time-Process": f"{t_proc - t_load:.3f}",
            "X-Time-Total": f"{t_proc - t0:.3f}",
        }
        return JSONResponse(content=resp, headers=headers)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))
    finally:
        # Cleanup after a delay to allow file serving
        async def cleanup():
            await asyncio.sleep(300)
            shutil.rmtree(tmp_dir, ignore_errors=True)
        asyncio.create_task(cleanup())

# ─── Download endpoint ─────────────────────────────────────
@app.post("/api/convert/download")
async def convert_download(
    file: UploadFile = File(...),
    with_text: bool = Form(True),
    with_refinement: bool = Form(False),
):
    """Convert and return the DrawIO XML file directly."""
    name = file.filename or "upload"
    ext = Path(name).suffix.lower()
    if ext not in ALLOWED_EXTS:
        raise HTTPException(400, f"Unsupported format '{ext}'")

    tmp_dir = tempfile.mkdtemp(prefix="eb_")
    tmp_path = os.path.join(tmp_dir, name)
    with open(tmp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        pipeline = await gpu_mgr.get_pipeline()
        output_dir = os.path.join(tmp_dir, "output")
        os.makedirs(output_dir, exist_ok=True)
        result_path = pipeline.process_image(
            tmp_path, output_dir=output_dir,
            with_refinement=with_refinement, with_text=with_text,
        )
        if not result_path or not os.path.exists(result_path):
            raise HTTPException(500, "Conversion failed")
        return FileResponse(
            result_path,
            media_type="application/xml",
            filename=Path(name).stem + ".drawio",
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))

# ─── GPU Management ────────────────────────────────────────
@app.post("/api/gpu-offload")
async def gpu_offload():
    ok = await gpu_mgr.offload()
    return {"success": ok, "gpu": gpu_mgr.gpu_info()}

@app.post("/api/gpu-reload")
async def gpu_reload():
    await gpu_mgr.get_pipeline()
    return {"success": True, "gpu": gpu_mgr.gpu_info()}

# ─── Static files ──────────────────────────────────────────
@app.get("/static/{path:path}")
async def static_file(path: str):
    fp = os.path.join(PROJECT_ROOT, "static", path)
    if os.path.isfile(fp):
        return FileResponse(fp)
    raise HTTPException(404, "Not found")

# ─── MCP SSE endpoint (proxy) ──────────────────────────────
MCP_PORT = int(os.getenv("MCP_PORT", "8452"))

# ─── Entrypoint ────────────────────────────────────────────
def main():
    port = int(os.getenv("PORT", "8450"))
    uvicorn.run(app, host="0.0.0.0", port=port, workers=1)

if __name__ == "__main__":
    main()
