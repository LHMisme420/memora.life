# ingest/transcribe.py
def transcribe_audio(audio_np, model):
    result = model.transcribe(audio_np, fp16=False, language="en")
    return result["text"].strip()
