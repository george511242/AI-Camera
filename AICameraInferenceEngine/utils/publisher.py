from classes.Subscribe import Publisher
from enum import Enum
class PublishEvent(str, Enum):
    UPDATED_CONFIG = 'UPDATED_CONFIG'
    UPDATED_SIGNATURE = 'UPDATED_SIGNATURE'

publisher = Publisher()