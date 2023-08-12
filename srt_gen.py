import whisper
import torch
from pathlib import Path
from whisper.utils import get_writer

print(torch.version)
print("USE GPU:", torch.cuda.is_available())

# GlobalVariable
Model_Type = "medium"
Data_File = "test.mp3"
save_filename = f"{Path(Data_File).stem}"

# check if you have a GPU available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#load Whipser model
model = whisper.load_model(Model_Type, device=DEVICE)
result = model.transcribe(Data_File)

# save TXT
txt_writer = get_writer("txt", ".")
txt_writer(result, save_filename)

# save SRT
srt_writer = get_writer("srt", ".")
srt_writer(result, save_filename)