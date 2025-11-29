# db/search.py
from db.init import table
from ingest.embed import embed_text

def recall(query: str, limit=10):
    vec = embed_text(query)
    results = table.search(vec).limit(limit).to_list()
    return [(r["distance"], r["timestamp"], r["type"], r["content"]) for r in results]
