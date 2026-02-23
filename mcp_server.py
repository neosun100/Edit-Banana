#!/usr/bin/env python3
"""Edit Banana MCP Server — expose convert/status/gpu tools via fastmcp SSE."""
import os, sys, json, tempfile, shutil, base64
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastmcp import FastMCP

mcp = FastMCP("edit-banana", description="Convert images/PDFs to editable DrawIO XML via SAM3 segmentation")

_pipeline = None

def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        from main import Pipeline, load_config
        _pipeline = Pipeline(load_config())
    return _pipeline

@mcp.tool()
def convert_image(image_path: str, with_text: bool = True, with_refinement: bool = False) -> dict:
    """Convert an image to editable DrawIO XML.
    Args:
        image_path: Absolute path to image file (png/jpg/bmp/webp/tiff)
        with_text: Enable OCR text extraction
        with_refinement: Enable quality refinement pass
    Returns:
        dict with xml_content, elements_count, output_path
    """
    if not os.path.exists(image_path):
        return {"error": f"File not found: {image_path}"}
    ext = Path(image_path).suffix.lower()
    if ext not in {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}:
        return {"error": f"Unsupported format: {ext}"}
    pipe = _get_pipeline()
    out_dir = tempfile.mkdtemp(prefix="eb_mcp_")
    result = pipe.process_image(image_path, output_dir=out_dir, with_text=with_text, with_refinement=with_refinement)
    if not result or not os.path.exists(result):
        return {"error": "Conversion failed"}
    with open(result, encoding="utf-8") as f:
        xml = f.read()
    return {"xml_content": xml, "elements_count": xml.count("<mxCell"), "output_path": result}

@mcp.tool()
def convert_pdf(pdf_path: str, with_text: bool = True) -> dict:
    """Convert a PDF to editable DrawIO XML (first page).
    Args:
        pdf_path: Absolute path to PDF file
        with_text: Enable OCR text extraction
    """
    if not os.path.exists(pdf_path):
        return {"error": f"File not found: {pdf_path}"}
    # PDF → image via Pillow/pdf2image
    try:
        from pdf2image import convert_from_path
        images = convert_from_path(pdf_path, first_page=1, last_page=1)
    except ImportError:
        return {"error": "pdf2image not installed"}
    if not images:
        return {"error": "Failed to convert PDF"}
    tmp = tempfile.mktemp(suffix=".png")
    images[0].save(tmp)
    result = convert_image(tmp, with_text=with_text)
    os.unlink(tmp)
    return result

@mcp.tool()
def get_status() -> dict:
    """Get service status including GPU info and model state."""
    import torch
    info = {"model_loaded": _pipeline is not None}
    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        total = torch.cuda.get_device_properties(idx).total_mem / 1e9
        used = torch.cuda.memory_allocated(idx) / 1e9
        info["gpu"] = {"device": torch.cuda.get_device_name(idx), "total_gb": round(total, 2), "used_gb": round(used, 2)}
    else:
        info["gpu"] = {"available": False}
    return info

@mcp.tool()
def gpu_offload() -> dict:
    """Release GPU memory by unloading models."""
    global _pipeline
    import torch, gc
    _pipeline = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"success": True, "message": "GPU memory released"}

if __name__ == "__main__":
    port = int(os.getenv("MCP_PORT", "8452"))
    mcp.run(transport="sse", host="0.0.0.0", port=port)
