FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV MCP_HOST=0.0.0.0
ENV PORT=8001
ENV MCP_PATH=/mcp

EXPOSE 8001

CMD ["python", "mcp_server.py"]
