#!/bin/bash
# 一键启动 repo-rag：pgvector + 服务（端口 8001），浏览器打开 http://localhost:8001
cd "$(dirname "$0")"
docker compose up -d db
exec uv run uvicorn app.main:app --port 8001
