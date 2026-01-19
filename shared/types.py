"""Shared type definitions for MCP servers."""

from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field


class MCPResponse(BaseModel):
    """Base MCP response."""

    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
