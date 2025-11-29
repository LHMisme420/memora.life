# capture/audio.py
import threading
import soundcard as sc
import whisper
from ingest.transcribe import transcribe_audio
from db.init import table
import time
import numpy as np

class AudioCapturer(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.running = True
        self.model = whisper.load_model("tiny")

    def run(self):
        with sc.get_microphone().recorder(samplerate=16000) as mic:
            while self.running:
                data = mic.record(numframes=16000*6)
                audio = np.squeeze(data)
                text = transcribe_audio(audio, self.model)
                if text and len(text) > 10:
                    from ingest.embed import embed_text
                    vec = embed_text(text)
                    table.add([{
                        "timestamp": time.time(),
                        "type": "audio",
                        "content": text,
                        "embedding": vec
                    }])
                    print(f"Audio: {text[:80]}...")

    def stop(self):
        self.running = False
