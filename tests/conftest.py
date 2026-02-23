"""Shared test fixtures for Edit Banana tests."""
import os
import sys
import io
import pytest
from unittest.mock import MagicMock, patch, AsyncMock

# Add project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture
def mock_pipeline():
    """Mock Pipeline that returns fake XML without GPU."""
    mock = MagicMock()
    mock.process_image.return_value = None  # Will be overridden per test
    return mock


@pytest.fixture
def sample_image_bytes():
    """Minimal valid PNG bytes."""
    from PIL import Image
    buf = io.BytesIO()
    img = Image.new("RGB", (100, 100), color="white")
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()


@pytest.fixture
def sample_xml():
    """Sample DrawIO XML output."""
    return '''<?xml version="1.0" encoding="UTF-8"?>
<mxfile>
  <diagram>
    <mxGraphModel>
      <root>
        <mxCell id="0"/>
        <mxCell id="1" parent="0"/>
        <mxCell id="2" parent="1" vertex="1" value="" style="rounded=0;">
          <mxGeometry x="10" y="10" width="80" height="60" as="geometry"/>
        </mxCell>
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>'''
