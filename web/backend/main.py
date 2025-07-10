"""
FastAPI backend for MCP Standards Server Web UI
"""

import json
import logging
import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from web.backend.engine_adapter import StandardsEngine

logger = logging.getLogger(__name__)


# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)


manager = ConnectionManager()


# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting MCP Standards Server Web UI")
    app.state.engine = StandardsEngine()
    await app.state.engine.initialize()
    yield
    # Shutdown
    logger.info("Shutting down MCP Standards Server Web UI")


# Create FastAPI app
app = FastAPI(
    title="MCP Standards Server",
    description="Web UI for browsing and interacting with development standards",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# API endpoints
@app.get("/")
async def root():
    return {"message": "MCP Standards Server API", "version": "1.0.0"}


@app.get("/api/standards")
async def get_all_standards():
    """Get all standards organized by category"""
    try:
        engine = app.state.engine
        standards = await engine.get_all_standards()

        # Organize by category
        by_category = {}
        for standard in standards:
            category = standard.category
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(
                {
                    "id": standard.id,
                    "title": standard.title,
                    "description": standard.description,
                    "tags": standard.tags,
                    "priority": standard.priority,
                    "version": standard.version,
                }
            )

        return {"standards": by_category, "total": len(standards)}
    except Exception as e:
        logger.error(f"Error fetching standards: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/standards/{standard_id}")
async def get_standard(standard_id: str):
    """Get detailed information about a specific standard"""
    try:
        engine = app.state.engine
        standard = await engine.get_standard_by_id(standard_id)
        if not standard:
            raise HTTPException(status_code=404, detail="Standard not found")

        return {
            "id": standard.id,
            "title": standard.title,
            "description": standard.description,
            "category": standard.category,
            "subcategory": standard.subcategory,
            "tags": standard.tags,
            "priority": standard.priority,
            "examples": standard.examples,
            "rules": standard.rules,
            "version": standard.version,
            "created_at": standard.created_at,
            "updated_at": standard.updated_at,
            "metadata": standard.metadata,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching standard {standard_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/search")
async def search_standards(query: dict[str, Any]):
    """Search standards with filters"""
    try:
        engine = app.state.engine
        search_query = query.get("query", "")
        filters = query.get("filters", {})
        limit = query.get("limit", 20)

        results = await engine.search_standards(
            query=search_query,
            category=filters.get("category"),
            tags=filters.get("tags"),
            limit=limit,
        )

        return {
            "results": [
                {
                    "id": r.standard.id,
                    "title": r.standard.title,
                    "description": r.standard.description,
                    "category": r.standard.category,
                    "score": r.score,
                    "highlights": r.highlights,
                }
                for r in results
            ],
            "total": len(results),
        }
    except Exception as e:
        logger.error(f"Error searching standards: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/analyze")
async def analyze_project(context: dict[str, Any]):
    """Analyze a project and get standard recommendations"""
    try:
        engine = app.state.engine

        # Create project context from request
        project_context = type("ProjectContext", (), context)()

        # Get recommendations
        recommendations = await engine.analyze_project(project_context)

        return {
            "recommendations": [
                {
                    "standard": {
                        "id": r.standard.id,
                        "title": r.standard.title,
                        "description": r.standard.description,
                        "category": r.standard.category,
                    },
                    "relevance_score": r.relevance_score,
                    "confidence": r.confidence,
                    "reasoning": r.reasoning,
                    "implementation_notes": r.implementation_notes,
                }
                for r in recommendations
            ],
            "total": len(recommendations),
        }
    except Exception as e:
        logger.error(f"Error analyzing project: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/categories")
async def get_categories():
    """Get all available categories"""
    try:
        engine = app.state.engine
        categories = await engine.get_categories()
        return {"categories": categories}
    except Exception as e:
        logger.error(f"Error fetching categories: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/tags")
async def get_tags():
    """Get all available tags"""
    try:
        engine = app.state.engine
        tags = await engine.get_tags()
        return {"tags": tags}
    except Exception as e:
        logger.error(f"Error fetching tags: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/export/bulk")
async def export_bulk_standards(request: dict[str, Any]):
    """Export multiple standards in various formats"""
    try:
        engine = app.state.engine
        standard_ids = request.get("standards", [])
        format = request.get("format", "json")

        if not standard_ids:
            # Export all standards if none specified
            all_standards = await engine.get_all_standards()
            standard_ids = [s.id for s in all_standards]

        standards_data = []
        for standard_id in standard_ids:
            standard = await engine.get_standard_by_id(standard_id)
            if standard:
                standards_data.append(
                    {
                        "id": standard.id,
                        "title": standard.title,
                        "description": standard.description,
                        "category": standard.category,
                        "subcategory": standard.subcategory,
                        "tags": standard.tags,
                        "priority": standard.priority,
                        "version": standard.version,
                        "examples": standard.examples,
                        "rules": standard.rules,
                        "metadata": standard.metadata,
                        "created_at": standard.created_at,
                        "updated_at": standard.updated_at,
                    }
                )

        if format == "json":
            export_data = {
                "exportDate": datetime.now().isoformat(),
                "totalStandards": len(standards_data),
                "standards": standards_data,
            }

            # Save to temporary file
            temp_file = Path(f"/tmp/standards-export-{datetime.now().timestamp()}.json")
            temp_file.write_text(json.dumps(export_data, indent=2))

            return FileResponse(
                path=str(temp_file),
                filename=f"standards-export-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json",
                media_type="application/json",
            )
        else:
            raise HTTPException(
                status_code=400, detail="Unsupported format for bulk export"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error bulk exporting standards: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/export/{standard_id}")
async def export_standard(standard_id: str, format: str = "markdown"):
    """Export a standard in various formats"""
    try:
        engine = app.state.engine
        standard = await engine.get_standard_by_id(standard_id)
        if not standard:
            raise HTTPException(status_code=404, detail="Standard not found")

        if format == "markdown":
            content = f"""# {standard.title}

**Category:** {standard.category}
**Subcategory:** {standard.subcategory}
**Priority:** {standard.priority}
**Version:** {standard.version}

## Description
{standard.description}

## Tags
{', '.join(standard.tags)}

## Rules
{json.dumps(standard.rules, indent=2)}

## Examples
"""
            for example in standard.examples:
                content += f"\n### {example.get('title', 'Example')}\n"
                content += f"```{example.get('language', '')}\n"
                content += f"{example.get('code', '')}\n"
                content += "```\n"
                if example.get("description"):
                    content += f"\n{example['description']}\n"

            # Save to temporary file
            temp_file = Path(f"/tmp/{standard_id}.md")
            temp_file.write_text(content)

            return FileResponse(
                path=str(temp_file),
                filename=f"{standard_id}.md",
                media_type="text/markdown",
            )
        elif format == "json":
            # Export as JSON
            export_data = {
                "id": standard.id,
                "title": standard.title,
                "description": standard.description,
                "category": standard.category,
                "subcategory": standard.subcategory,
                "tags": standard.tags,
                "priority": standard.priority,
                "version": standard.version,
                "examples": standard.examples,
                "rules": standard.rules,
                "metadata": standard.metadata,
                "created_at": standard.created_at,
                "updated_at": standard.updated_at,
            }

            temp_file = Path(f"/tmp/{standard_id}.json")
            temp_file.write_text(json.dumps(export_data, indent=2))

            return FileResponse(
                path=str(temp_file),
                filename=f"{standard_id}.json",
                media_type="application/json",
            )
        else:
            raise HTTPException(
                status_code=400,
                detail="Unsupported format. Supported formats: markdown, json",
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting standard {standard_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# WebSocket endpoint for real-time updates
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message = json.loads(data)

            # Handle different message types
            if message["type"] == "ping":
                await manager.send_personal_message(
                    json.dumps(
                        {"type": "pong", "timestamp": datetime.now().isoformat()}
                    ),
                    websocket,
                )
            elif message["type"] == "subscribe":
                # Subscribe to updates for specific standards or categories
                await manager.send_personal_message(
                    json.dumps(
                        {
                            "type": "subscribed",
                            "to": message.get("to", "all"),
                            "timestamp": datetime.now().isoformat(),
                        }
                    ),
                    websocket,
                )

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info("Client disconnected from WebSocket")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=os.getenv("WEB_HOST", "127.0.0.1"), port=8000)
