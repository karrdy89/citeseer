import torch
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer
from adapters import AutoAdapterModel


model = None
tokenizer = None
device = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer, device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")
    model = AutoAdapterModel.from_pretrained("allenai/specter2_base")
    model.load_adapter(
        "allenai/specter2_adhoc_query",
        source="hf", load_as="adhoc_query", set_active=True,
    )
    model.to(device).eval()
    print(f"SPECTER2 loaded on {device}")
    yield


app = FastAPI(lifespan=lifespan)


class EncodeRequest(BaseModel):
    text: str | list[str]


class EncodeResponse(BaseModel):
    embeddings: list[list[float]]


@app.post("/encode", response_model=EncodeResponse)
async def encode(req: EncodeRequest):
    texts = req.text if isinstance(req.text, list) else [req.text]
    inputs = tokenizer(
        texts, padding=True, truncation=True,
        return_tensors="pt", max_length=512,
    ).to(device)
    with torch.no_grad():
        output = model(**inputs)
    vecs = output.last_hidden_state[:, 0, :].cpu().tolist()
    return EncodeResponse(embeddings=vecs)


@app.get("/health")
async def health():
    return {"status": "ok", "device": str(device)}
