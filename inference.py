import os

import soundfile as sf
import torch
import torchaudio
from transformers import AutoModelForCTC, Wav2Vec2BertProcessor

# Load model and processor
model_name = "atabekm/w2v-bert-2.0-karakalpak-colab"
processor = Wav2Vec2BertProcessor.from_pretrained(model_name)
model = AutoModelForCTC.from_pretrained(model_name).to("cpu").eval()


def load_audio(file_path):
  audio, sample_rate = sf.read(file_path, dtype="float32")
  audio_tensor = torch.tensor(audio, dtype=torch.float32)
  return audio_tensor, sample_rate


def preprocess_audio(file_path):
  wf, sample_rate = load_audio(file_path)

  # Convert to 16 kHz if needed
  if sample_rate != 16000:
    transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
    wf = transform(wf)

  return wf.squeeze()


audio_dir = "./wav"  # Change to your directory containing WAV files
audio_files = [os.path.join(audio_dir, f) for f in os.listdir(audio_dir) if f.endswith(".wav")]
audio_files.sort()

transcriptions = {}

for file_path in audio_files:
  waveform = preprocess_audio(file_path)

  # Tokenize input
  inputs = processor(waveform, sampling_rate=16000, return_tensors="pt", padding=True)

  # Run inference
  with torch.no_grad():
    logits = model(inputs["input_features"]).logits

  # Decode predictions
  predicted_ids = torch.argmax(logits, dim=-1)
  transcription = processor.batch_decode(predicted_ids)[0]

  transcriptions[file_path] = transcription

# Print all transcriptions
for file, text in transcriptions.items():
  print(f"{file}: {text}")
