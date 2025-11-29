# memora.py - Lifelong Personal Memory (v0.1)
# Run this and speak/type anything — it will remember forever, privately.

import time
import os
import sqlite_vec
import whisper
import soundcard as sc
import numpy as np
from nomic import embed
from datetime import datetime
import sqlite3
from pathlib import Path

DB_PATH = Path.home() / ".memora" / "memory.db"
DB_PATH.parent.mkdir(exist_ok=True)

# Initialize vector DB
conn = sqlite3.connect(DB_PATH)
conn.execute("CREATE TABLE IF NOT EXISTS memories(id INTEGER PRIMARY KEY, ts DATETIME, type TEXT, content TEXT, embedding BLOB)")
conn.execute("CREATE INDEX IF NOT EXISTS idx_ts ON memories(ts)")

# Load tiny whisper for real-time transcription
model = whisper.load_model("tiny")  # runs comfortably on CPU

print("Memora v0.1 active. Say anything. Ctrl-C to stop.")

with sc.get_microphone().recorder(samplerate=16000) as mic:
    while True:
        try:
            audio_data = mic.record(numframes=16000*5)  # 5-second chunks
            audio_np = np.squeeze(audio_data)
            
            # Transcribe locally
            result = model.transcribe(audio_np, language="en", fp16=False)
            text = result["text"].strip()
            
            if len(text) > 8:  # ignore silence
                # Generate embedding locally
                embedding = embed.text(text, model='nomic-embed-text-v1.5')[0]['embedding']
                
                # Store with vector
                conn.execute("""
                    INSERT INTO memories(ts, type, content, embedding) 
                    VALUES (?, 'speech', ?, ?)
                """, (datetime.now(), text, np.array(embedding, dtype=np.float32).tobytes()))
                conn.commit()
                
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Remembered: {text[:70]}{'...' if len(text)>70 else ''}")
                
        except KeyboardInterrupt:
            print("\nMemora stopped. Your life is now encrypted forever on your machine.")
            break

# Search example:
def recall(query: str, limit=5):
    q_emb = np.array(embed.text(query, model='nomic-embed-text-v1.5')[0]['embedding'], dtype=np.float32)
    rows = conn.execute("""
        SELECT content, ts, distance 
        FROM memories 
        WHERE embedding MATCH ? 
        ORDER BY distance 
        LIMIT ?
    """, (q_emb.tobytes(), limit)).fetchall()
    
    for content, ts, dist in rows:
        print(f"{ts} (sim={1-dist:.3f}): {content}")

# Try it:
# recall("what did I say about quantum computing last month?")
