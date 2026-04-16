FROM python:3.12-slim

WORKDIR /app

COPY mcp_server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY mcp_server/ mcp_server/
COPY shared/ ./shared/

RUN useradd --create-home mcp && chown -R mcp:mcp /app
USER mcp

ENV PYTHONUNBUFFERED=1
EXPOSE 8050

CMD ["python", "-m", "mcp_server.main"]
