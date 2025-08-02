from pvrecorder import PvRecorder
from pvspeaker import PvSpeaker 

print("Input Devices:")
for i, device in enumerate(PvRecorder.get_available_devices()):
  print(f"{i}. {device}")

print("\nOutput Devices:")
for i, device in enumerate(PvSpeaker.get_available_devices()):
  print(f"{i}. {device}")