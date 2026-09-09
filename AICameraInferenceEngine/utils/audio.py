import subprocess
import json


def get_device_usb_audio():
  try:
    cmd = "cat /proc/asound/cards | grep -i usb | cut -d' ' -f2 | tr -d '[] \n' | sed 's/./&,/g' | sed 's/,$//' | sed 's/^/[/' | sed 's/$/]/' || echo '[]'"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode == 0:
      output = result.stdout.strip()
      if output:
        return json.loads(output)
      else:
        return []
    else:
      return []
  except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
    return []

def get_device_usb_audio_volume(index: int):
  # amixer -c 2 get PCM | grep -o '[0-9]*%' | head -1 | sed 's/%//'
  try:
    # Get the list of USB audio devices
    usb_audio_devices = get_device_usb_audio()
    # Check if the index is valid
    if index not in usb_audio_devices:
      raise ValueError("Invalid device index")
    # Get the volume for the specified USB audio device
    cmd = f"amixer -c {index} get PCM | grep -o '[0-9]*%' | head -1 | sed 's/%//'"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode == 0:
      return int(result.stdout.strip())
    else:
      raise RuntimeError("Failed to get volume")
  except subprocess.CalledProcessError as e:
    raise RuntimeError(f"Failed to get volume: {e}")
  except ValueError as e:
    raise ValueError(f"Invalid device index: {e}")


def set_device_usb_audio_volume(index: int, volume: int):
  # amixer -c 2 set PCM 50%
  try:
    # Get the list of USB audio devices
    usb_audio_devices = get_device_usb_audio()
    # Check if the index is valid
    if index not in usb_audio_devices:
      raise ValueError("Invalid device index")
    # Set the volume for the specified USB audio device
    cmd = f"amixer -c {index} set PCM {volume}% && alsactl store"
    subprocess.run(cmd, shell=True, check=True)
  except subprocess.CalledProcessError as e:
    raise RuntimeError(f"Failed to set volume: {e}")
  except ValueError as e:
    raise ValueError(f"Invalid device index: {e}")

def restore_usb_audio_volume():
  try:
    cmd = "alsactl restore"
    subprocess.run(cmd, shell=True, check=True)
  except subprocess.CalledProcessError as e:
    raise RuntimeError(f"Failed to restore volume: {e}")