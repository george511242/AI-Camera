from pydantic import BaseModel, Field
from datetime import datetime
from typing import Union

class StreamingRequestSchema(BaseModel):
  source: Union[str, int]

class RecordRangeSchema(BaseModel):
  start_time: float
  end_time: float

class StreamRecordRequestSchema(BaseModel):
  post_url: str = Field(default='')
  range: RecordRangeSchema = Field(default=RecordRangeSchema(start_time=0, end_time=0))
  split_time: int = Field(default=0)