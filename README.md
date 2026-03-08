# RetailNext Luxury Associate Assistant

This is a Streamlit app for a luxury retail associate workflow.

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/enoshneupane/retaildemo)

## Requirements

- Python 3.11

## Setup

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run

```bash
source .venv/bin/activate
python -m streamlit run app.py
```

## MCP Server

This repo also includes a remote-ready MCP server for ChatGPT in `mcp_server.py`.

For ChatGPT uploads, the model should inspect the uploaded dress image and call the chat-image matching tool. The MCP server itself does not read chat attachments directly.
The dress-match tools now expose a ChatGPT app view that renders the recommended dresses as an image card carousel with product details.

Run it locally for testing:

```bash
source .venv/bin/activate
python mcp_server.py
```

The MCP endpoint is:

```text
http://localhost:8001/mcp
```

ChatGPT requires a remotely hosted MCP server URL, so deploy this server to a public host before adding it to ChatGPT.

### Deploy On Render

1. Push this folder to a GitHub repo.
2. In Render, create a new `Blueprint` and select that repo.
3. Render will pick up [`render.yaml`](./render.yaml) automatically.
4. After deploy, your MCP endpoint will be:

```text
https://your-render-service.onrender.com/mcp
```

5. Add that URL as a custom MCP app/connector in ChatGPT.

For testing, Render `free` works. For actual ChatGPT use, `starter` is recommended to avoid cold-start delays.

You can also run the MCP server in Docker:

```bash
docker build -t retailnext-mcp .
docker run -p 8001:8001 retailnext-mcp
```

After deployment, use the public MCP endpoint in this format:

```text
https://your-domain.example/mcp
```

## Input

Upload `gala_inspiration.jpg` from `images/uploads/` to begin the styling flow.
