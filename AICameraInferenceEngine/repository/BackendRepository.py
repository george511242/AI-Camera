import requests
from settings import settings
class BackendRepository():
    
    def get_register_signature(self):
        response = requests.get(f'{settings.backend_url}/wms/signature', headers={ 'x-token': 'auo_ai_camera' })
        return response.text
    
    def restart_daemon(self):
        try:
            return requests.post(f'{settings.backend_url}/api/v1/device/daemon/restart', headers={ 'x-token': 'auo_ai_camera' }, timeout=60)
        except Exception as e:
            logger.error(e)
            return { "success": False, "detail": "Unable to restart daemon" }