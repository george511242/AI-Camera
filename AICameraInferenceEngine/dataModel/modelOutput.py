from typing import Optional, Any
from pydantic import BaseModel, Field, field_validator
from enum import Enum
import uuid

class InferenceMode(str, Enum):
  LeaveZone = 'leave-zone'
  LeaveZoneWithAgeGender = 'leave-zone-with-age-gender'
  EyeTracking = 'leave-zone-with-eye-tracking'
  LeaveZoneWithFacialFeatures = 'leave-zone-with-facial-features'
  PhonePoseDetection = 'phone-pose-detection'
  PeopleCounting = 'people-counting'

class S4MRecordSchema(BaseModel):
    regionId: str = Field(default='', alias='region')  # 關注區 ID
    personId: str = Field(default='', alias='id')  # 追蹤演算法紀錄的 person ID
    enterDate: int = Field(default=-1, alias='enter_date')  # 進入關注區的 timestamp (秒)
    leaveDate: int = Field(default=-1, alias='leave_date')  # 離開關注區的 timestamp (秒)
    startDate: int = Field(default=-1, alias='start_date')  # 開始看鏡頭的 timestamp (秒)
    endDate: int = Field(default=-1, alias='end_date')  # 結束看鏡頭的 timestamp (秒)
    eventId: str = Field(default_factory=lambda: uuid.uuid4().hex)
    mode: str = Field(default='')
    age: int = Field(default=-1)
    gender: int = Field(default=-1)
    pitch: int = Field(default=180)
    roll: int = Field(default=180)
    yaw: int = Field(default=180)
    snapshot: Optional[Any] = Field(default=None)

    @field_validator('age', 'gender', 'pitch', 'roll', 'yaw', mode='before')
    @classmethod
    def to_int(cls, raw: Any) -> int:
        return int(raw)

class PhonePoseSchema(BaseModel):
    eventId: str = Field(default_factory=lambda: uuid.uuid4().hex)
    personId: str = Field(default='', alias='id')
    startDate: int = Field(default=-1, alias='start_date')
    snapshot: Optional[Any] = Field(default=None)

class PeopleCountSchema(BaseModel):
    start_ts: int = Field(default=-1)  # 開始時間戳 (秒)
    end_ts: int = Field(default=-1)    # 結束時間戳 (秒)
    id_count: int = Field(default=0)      # 人數統計
    mode: InferenceMode = Field(default=InferenceMode.PeopleCounting)  # Mode

class S4MInteractiveSchema(BaseModel):
    # TODO: check output fileds
    eventId: str = Field(default_factory=lambda: uuid.uuid4().hex)  # uuid
