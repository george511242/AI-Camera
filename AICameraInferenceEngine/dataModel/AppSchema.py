from typing import Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime

class HealthSchema(BaseModel):
  health: bool
  error: Optional[Any] = None