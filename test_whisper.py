import whisper
import torch

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

model=whisper.load_model("base").to("cuda")

print("Whisper loaded on GPU successfully")
