"""
Full API E2E tests for Edit Banana.
Tests every endpoint, error cases, parameter variants, concurrency, CORS.
Uses mock pipeline (no GPU required).
"""
import io
import os
import sys
import json
import tempfile
import concurrent.futures
from unittest.mock import patch, MagicMock, AsyncMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Mock torch before importing app
mock_torch = MagicMock()
mock_torch.cuda.is_available.return_value = False
mock_torch.cuda.empty_cache = MagicMock()

SAMPLE_XML = '''<?xml version="1.0" encoding="UTF-8"?>
<mxfile><diagram><mxGraphModel><root>
<mxCell id="0"/><mxCell id="1" parent="0"/>
<mxCell id="2" parent="1" vertex="1" value="" style="rounded=0;">
  <mxGeometry x="10" y="10" width="80" height="60" as="geometry"/>
</mxCell>
<mxCell id="3" parent="1" vertex="1" value="" style="ellipse;">
  <mxGeometry x="100" y="100" width="50" height="50" as="geometry"/>
</mxCell>
</root></mxGraphModel></diagram></mxfile>'''


def _make_png():
    """Create minimal PNG bytes."""
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (100, 100), "white").save(buf, format="PNG")
    buf.seek(0)
    return buf.read()


def _mock_pipeline_process(image_path, output_dir=None, with_refinement=False, with_text=True, groups=None):
    """Mock pipeline that writes XML to output_dir and returns path."""
    if output_dir is None:
        output_dir = tempfile.mkdtemp()
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "result.drawio")
    with open(out_path, "w") as f:
        f.write(SAMPLE_XML)
    return out_path


@pytest.fixture(autouse=True)
def patch_gpu_and_pipeline():
    """Patch GPU manager to use mock pipeline."""
    with patch.dict("sys.modules", {"torch": mock_torch}):
        # Import app after patching
        import importlib
        if "app" in sys.modules:
            importlib.reload(sys.modules["app"])

        from app import app, gpu_mgr

        # Mock the pipeline
        mock_pipe = MagicMock()
        mock_pipe.process_image.side_effect = _mock_pipeline_process
        mock_pipe.config = {"sam3": {"score_threshold": 0.5, "epsilon_factor": 0.02, "min_area": 100}}
        gpu_mgr.pipeline = mock_pipe
        gpu_mgr.sam3_loaded = True

        from fastapi.testclient import TestClient
        client = TestClient(app)
        yield client, gpu_mgr


# ==================== Health & Status ====================

class TestHealth:
    def test_health_200(self, patch_gpu_and_pipeline):
        client, _ = patch_gpu_and_pipeline
        r = client.get("/health")
        assert r.status_code == 200
        d = r.json()
        assert d["status"] == "healthy"
        assert "version" in d
        assert "gpu" in d
        assert "model_loaded" in d

    def test_health_has_version(self, patch_gpu_and_pipeline):
        client, _ = patch_gpu_and_pipeline
        d = client.get("/health").json()
        assert d["version"] == "1.0.0"

    def test_status_200(self, patch_gpu_and_pipeline):
        client, _ = patch_gpu_and_pipeline
        r = client.get("/api/status")
        assert r.status_code == 200
        d = r.json()
        assert "service" in d
        assert "supported_formats" in d
        assert ".png" in d["supported_formats"]

    def test_status_has_gpu_info(self, patch_gpu_and_pipeline):
        client, _ = patch_gpu_and_pipeline
        d = client.get("/api/status").json()
        assert "gpu" in d

    def test_docs_accessible(self, patch_gpu_and_pipeline):
        client, _ = patch_gpu_and_pipeline
        r = client.get("/docs")
        assert r.status_code == 200

    def test_openapi_json(self, patch_gpu_and_pipeline):
        client, _ = patch_gpu_and_pipeline
        r = client.get("/openapi.json")
        assert r.status_code == 200
        spec = r.json()
        assert "paths" in spec
        assert "/health" in spec["paths"]


# ==================== UI ====================

class TestUI:
    def test_ui_loads(self, patch_gpu_and_pipeline):
        client, _ = patch_gpu_and_pipeline
        r = client.get("/")
        assert r.status_code == 200
        assert "Edit Banana" in r.text

    def test_ui_has_dark_theme(self, patch_gpu_and_pipeline):
        client, _ = patch_gpu_and_pipeline
        r = client.get("/")
        assert "#0d1117" in r.text or "dark" in r.text.lower()

    def test_ui_has_i18n(self, patch_gpu_and_pipeline):
        client, _ = patch_gpu_and_pipeline
        r = client.get("/")
        for lang in ["en", "zh", "tw", "ja"]:
            assert lang in r.text


# ==================== Config ====================

class TestConfig:
    def test_get_config(self, patch_gpu_and_pipeline):
        client, _ = patch_gpu_and_pipeline
        r = client.get("/api/config")
        assert r.status_code == 200
        d = r.json()
        # Either config data or error (if file missing in test env)
        assert isinstance(d, dict)


# ==================== Convert API ====================

class TestConvert:
    def test_convert_png(self, patch_gpu_and_pipeline):
        client, _ = patch_gpu_and_pipeline
        png = _make_png()
        r = client.post("/api/convert",
            files={"file": ("test.png", io.BytesIO(png), "image/png")},
            data={"with_text": "true", "with_refinement": "false"})
        assert r.status_code == 200
        d = r.json()
        assert d["success"] is True
        assert d["elements_count"] > 0
        assert "xml_content" in d
        assert "mxCell" in d["xml_content"]

    def test_convert_jpg(self, patch_gpu_and_pipeline):
        client, _ = patch_gpu_and_pipeline
        from PIL import Image
        buf = io.BytesIO()
        Image.new("RGB", (50, 50), "red").save(buf, format="JPEG")
        buf.seek(0)
        r = client.post("/api/convert",
            files={"file": ("test.jpg", buf, "image/jpeg")},
            data={"with_text": "false"})
        assert r.status_code == 200
        assert r.json()["success"]

    def test_convert_with_params(self, patch_gpu_and_pipeline):
        client, _ = patch_gpu_and_pipeline
        png = _make_png()
        r = client.post("/api/convert",
            files={"file": ("test.png", io.BytesIO(png), "image/png")},
            data={"score_threshold": "0.3", "epsilon_factor": "0.01", "min_area": "50"})
        assert r.status_code == 200

    def test_convert_timing_headers(self, patch_gpu_and_pipeline):
        client, _ = patch_gpu_and_pipeline
        png = _make_png()
        r = client.post("/api/convert",
            files={"file": ("test.png", io.BytesIO(png), "image/png")})
        assert r.status_code == 200
        assert "x-time-total" in r.headers

    def test_convert_no_refinement(self, patch_gpu_and_pipeline):
        client, _ = patch_gpu_and_pipeline
        png = _make_png()
        r = client.post("/api/convert",
            files={"file": ("test.png", io.BytesIO(png), "image/png")},
            data={"with_refinement": "false"})
        assert r.status_code == 200

    def test_convert_with_refinement(self, patch_gpu_and_pipeline):
        client, _ = patch_gpu_and_pipeline
        png = _make_png()
        r = client.post("/api/convert",
            files={"file": ("test.png", io.BytesIO(png), "image/png")},
            data={"with_refinement": "true"})
        assert r.status_code == 200


# ==================== Error Handling ====================

class TestErrors:
    def test_unsupported_format(self, patch_gpu_and_pipeline):
        client, _ = patch_gpu_and_pipeline
        r = client.post("/api/convert",
            files={"file": ("test.xyz", io.BytesIO(b"fake"), "application/octet-stream")})
        assert r.status_code == 400
        d = r.json()
        assert "error" in d

    def test_no_file(self, patch_gpu_and_pipeline):
        client, _ = patch_gpu_and_pipeline
        r = client.post("/api/convert")
        assert r.status_code == 422

    def test_empty_file(self, patch_gpu_and_pipeline):
        client, _ = patch_gpu_and_pipeline
        r = client.post("/api/convert",
            files={"file": ("empty.png", io.BytesIO(b""), "image/png")})
        # Should either 400 or 500 with JSON error
        assert r.status_code in [400, 500, 200]
        if r.status_code != 200:
            assert "error" in r.json() or "detail" in r.json()

    def test_404_unknown_path(self, patch_gpu_and_pipeline):
        client, _ = patch_gpu_and_pipeline
        r = client.get("/nonexistent/path")
        assert r.status_code == 404

    def test_error_returns_json_not_html(self, patch_gpu_and_pipeline):
        client, _ = patch_gpu_and_pipeline
        r = client.post("/api/convert",
            files={"file": ("test.txt", io.BytesIO(b"hello"), "text/plain")})
        assert r.status_code in [400, 422]
        assert r.headers.get("content-type", "").startswith("application/json")

    def test_invalid_score_threshold(self, patch_gpu_and_pipeline):
        client, _ = patch_gpu_and_pipeline
        png = _make_png()
        r = client.post("/api/convert",
            files={"file": ("test.png", io.BytesIO(png), "image/png")},
            data={"score_threshold": "not_a_number"})
        assert r.status_code == 422


# ==================== GPU Management ====================

class TestGPU:
    def test_gpu_offload(self, patch_gpu_and_pipeline):
        client, mgr = patch_gpu_and_pipeline
        r = client.post("/api/gpu-offload")
        assert r.status_code == 200
        d = r.json()
        assert "success" in d

    def test_gpu_reload(self, patch_gpu_and_pipeline):
        client, mgr = patch_gpu_and_pipeline
        r = client.post("/api/gpu-reload")
        assert r.status_code == 200
        assert r.json()["success"]

    def test_offload_then_convert(self, patch_gpu_and_pipeline):
        """GPU offload then convert should auto-reload."""
        client, mgr = patch_gpu_and_pipeline
        client.post("/api/gpu-offload")
        # Re-mock pipeline for next call
        mock_pipe = MagicMock()
        mock_pipe.process_image.side_effect = _mock_pipeline_process
        mock_pipe.config = {"sam3": {}}

        async def fake_get():
            mgr.pipeline = mock_pipe
            mgr.sam3_loaded = True
            return mock_pipe

        mgr.get_pipeline = fake_get
        png = _make_png()
        r = client.post("/api/convert",
            files={"file": ("test.png", io.BytesIO(png), "image/png")})
        assert r.status_code == 200


# ==================== CORS ====================

class TestCORS:
    def test_cors_headers(self, patch_gpu_and_pipeline):
        client, _ = patch_gpu_and_pipeline
        r = client.options("/api/convert",
            headers={"Origin": "http://example.com", "Access-Control-Request-Method": "POST"})
        header_keys = {k.lower() for k in r.headers.keys()}
        assert "access-control-allow-origin" in header_keys


# ==================== Concurrency ====================

class TestConcurrency:
    def test_concurrent_health(self, patch_gpu_and_pipeline):
        client, _ = patch_gpu_and_pipeline
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(lambda: client.get("/health")) for _ in range(10)]
            results = [f.result() for f in futures]
        assert all(r.status_code == 200 for r in results)

    def test_concurrent_status(self, patch_gpu_and_pipeline):
        client, _ = patch_gpu_and_pipeline
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(lambda: client.get("/api/status")) for _ in range(5)]
            results = [f.result() for f in futures]
        assert all(r.status_code == 200 for r in results)


# ==================== Parameter Variants ====================

class TestParamVariants:
    @pytest.mark.parametrize("score", [0.1, 0.3, 0.5, 0.7, 0.9])
    def test_score_threshold_variants(self, patch_gpu_and_pipeline, score):
        client, _ = patch_gpu_and_pipeline
        png = _make_png()
        r = client.post("/api/convert",
            files={"file": ("test.png", io.BytesIO(png), "image/png")},
            data={"score_threshold": str(score)})
        assert r.status_code == 200

    @pytest.mark.parametrize("eps", [0.005, 0.01, 0.02, 0.05])
    def test_epsilon_variants(self, patch_gpu_and_pipeline, eps):
        client, _ = patch_gpu_and_pipeline
        png = _make_png()
        r = client.post("/api/convert",
            files={"file": ("test.png", io.BytesIO(png), "image/png")},
            data={"epsilon_factor": str(eps)})
        assert r.status_code == 200

    @pytest.mark.parametrize("area", [10, 50, 100, 500, 1000])
    def test_min_area_variants(self, patch_gpu_and_pipeline, area):
        client, _ = patch_gpu_and_pipeline
        png = _make_png()
        r = client.post("/api/convert",
            files={"file": ("test.png", io.BytesIO(png), "image/png")},
            data={"min_area": str(area)})
        assert r.status_code == 200

    @pytest.mark.parametrize("text,refine", [(True, False), (False, False), (True, True), (False, True)])
    def test_boolean_combos(self, patch_gpu_and_pipeline, text, refine):
        client, _ = patch_gpu_and_pipeline
        png = _make_png()
        r = client.post("/api/convert",
            files={"file": ("test.png", io.BytesIO(png), "image/png")},
            data={"with_text": str(text).lower(), "with_refinement": str(refine).lower()})
        assert r.status_code == 200
