# Edit Banana MCP Guide

## Connection

Edit Banana exposes an MCP server via SSE on port 8452.

### Configuration

Add to your MCP client config:

```json
{
  "mcpServers": {
    "edit-banana": {
      "url": "http://localhost:8452/sse"
    }
  }
}
```

## Available Tools

| Tool | Description |
|------|-------------|
| `convert_image` | Convert image to DrawIO XML |
| `convert_pdf` | Convert PDF (first page) to DrawIO XML |
| `get_status` | Get GPU/model status |
| `gpu_offload` | Release GPU memory |

## Examples

### Convert an image
```
convert_image(image_path="/path/to/flowchart.png", with_text=True)
```

### Check status
```
get_status()
```
