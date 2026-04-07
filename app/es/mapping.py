"""ES papers 인덱스 매핑."""

PAPERS_MAPPING = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
    },
    "mappings": {
        "properties": {
            "doi":            {"type": "keyword"},
            "arxiv_id":       {"type": "keyword"},
            "s2_id":          {"type": "keyword"},
            "title":          {"type": "text"},
            "abstract":       {"type": "text"},
            "authors":        {"type": "keyword"},
            "year":           {"type": "integer"},
            "publication_date": {"type": "date", "format": "yyyy-MM-dd"},
            "venue":          {"type": "keyword"},
            "citation_count": {"type": "integer"},
            "tldr":           {"type": "text"},
            "keywords":       {"type": "keyword"},
            "content_type":   {"type": "keyword"},
            "pdf_path":       {"type": "keyword", "index": False},
            "json_path":      {"type": "keyword", "index": False},
            "url":            {"type": "keyword", "index": False},
            "embedding": {
                "type": "dense_vector",
                "dims": 768,
                "index": True,
                "similarity": "cosine",
            },
            "ingested_at":    {"type": "date"},
            "updated_at":     {"type": "date"},
        }
    },
}
