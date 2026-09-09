from repository.ConfigRepository import ConfigRepository
from utils.publisher import publisher, PublishEvent

config = ConfigRepository()

def reload_config():
    config.reload_config()
    publisher.publish(PublishEvent.UPDATED_CONFIG)
    
def set_config(category, key: str, value: str):
    config.set_config(category, key, value)

def save_config():
    config.save()
        
def get_config(category, key, default = None):
    return config.get_config(category, key, default)