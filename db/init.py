# db/init.py
import lancedb
from pathlib import Path
import sqlite3

DB_PATH = Path.home() / ".memora" / "memories.lance"
DB_PATH.parent.mkdir(exist_ok=True)

db = lancedb.connect(str(DB_PATH.parent))

if "memories" not in db.table_names():
    db.create_table("memories", schema={
        "id": "int",
        "timestamp": "float",
        "type": "string",  # audio, screen, webcam, note
        "content": "string",
        "embedding": "vector(768)"
    })
    print("Memory database created.")

table = db.open_table("memories")
