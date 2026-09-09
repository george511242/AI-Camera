from utils.readConfig import config, get_config

class QueueConfig:
  @property
  def max_size(self):
    return int(get_config('QUEUE', 'max_size', 50000))
  
  @property
  def max_days(self):
    return int(get_config('QUEUE', 'max_days', 7))