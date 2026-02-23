"""
MCP server tool tests for Edit Banana.
Tests the tool functions directly (without fastmcp transport layer).
"""
import os
import sys
import tempfile
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Mock torch and fastmcp before importing
mock_torch = MagicMock()
mock_torch.cuda.is_available.return_value = False
mock_torch.cuda.empty_cache = MagicMock()

mock_fastmcp = MagicMock()
mock_mcp_instance = MagicMock()
mock_fastmcp.return_value = mock_mcp_instance
# Make @mcp.tool() a passthrough decorator
mock_mcp_instance.tool.return_value = lambda fn: fn

SAMPLE_XML = '<mxfile><diagram><mxGraphModel><root><mxCell id="0"/><mxCell id="1" parent="0"/><mxCell id="2" parent="1" vertex="1"/></root></mxGraphModel></diagram></mxfile>'


@pytest.fixture(autouse=True)
def mock_deps():
    with patch.dict("sys.modules", {
        "torch": mock_torch,
        "fastmcp": MagicMock(FastMCP=mock_fastmcp),
    }):
        # Force reimport
        if "mcp_server" in sys.modules:
            del sys.modules["mcp_server"]
        yield


class TestMCPTools:
    def test_convert_image_file_not_found(self, mock_deps):
        import mcp_server
        result = mcp_server.convert_image("/nonexistent/file.png")
        assert "error" in result

    def test_convert_image_unsupported_format(self, mock_deps):
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"hello")
            path = f.name
        try:
            import mcp_server
            result = mcp_server.convert_image(path)
            assert "error" in result
        finally:
            os.unlink(path)

    def test_convert_image_with_mock(self, mock_deps):
        from PIL import Image
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            Image.new("RGB", (100, 100), "white").save(f, format="PNG")
            path = f.name

        mock_pipe = MagicMock()
        out_dir = tempfile.mkdtemp()
        out_path = os.path.join(out_dir, "result.drawio")
        with open(out_path, "w") as f:
            f.write(SAMPLE_XML)
        mock_pipe.process_image.return_value = out_path

        try:
            import mcp_server
            mcp_server._pipeline = mock_pipe
            result = mcp_server.convert_image(path)
            assert "xml_content" in result
            assert result["elements_count"] > 0
        finally:
            os.unlink(path)
            mcp_server._pipeline = None

    def test_get_status(self, mock_deps):
        import mcp_server
        result = mcp_server.get_status()
        assert "model_loaded" in result
        assert "gpu" in result

    def test_gpu_offload(self, mock_deps):
        import mcp_server
        result = mcp_server.gpu_offload()
        assert result["success"] is True

    def test_convert_pdf_missing_file(self, mock_deps):
        import mcp_server
        result = mcp_server.convert_pdf("/nonexistent/file.pdf")
        assert "error" in result

    def test_status_after_offload(self, mock_deps):
        import mcp_server
        mcp_server.gpu_offload()
        status = mcp_server.get_status()
        assert status["model_loaded"] is False
