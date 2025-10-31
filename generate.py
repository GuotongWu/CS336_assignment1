import os
from pathlib import Path

import torch
import uvicorn
import wandb
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from cs336_basics.bpe import BPE
from cs336_basics.transformer import Transformer

load_dotenv()
api = wandb.Api(
    overrides={"base_url": os.getenv("WANDB_BASE_URL")}, 
    api_key=os.getenv("WANDB_API_KEY"))
sweep = api.sweep("wuguotongswjtu-southern-university-of-science-technology/CS336_assignment1/fwrqlh17")
best_run = sweep.best_run()
print(best_run.name)

args = best_run.config

checkpoint_path = os.path.join(
    args["checkpoint_path"],
    f"{args['dataset_name']}_maxlr{args['max_lr']:.6f}_batch{args['batch_size']}.pth",
)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = Transformer(
    vocab_size=args["vocab_size"],
    context_length=args["context_length"],
    num_layers=args["num_layers"],
    d_model=args["d_model"],
    num_heads=args["num_heads"],
    d_ff=args["d_ff"],
    rope_theta=args["rope_theta"],
    device=device,
)
state_dict = torch.load(checkpoint_path, map_location=device)["model_state_dict"]
model.load_state_dict(state_dict)
model.to(device)
model.eval()

vocab_merges_filepath = "data/TinyStoriesV2-GPT4-train-vocab-merges-10000.pkl"

SPECIAL_TOKENS = ["<|endoftext|>"]
tokenizer = BPE.from_vocab_merges_files(vocab_merges_filepath, SPECIAL_TOKENS)

WEB_DIR = Path(__file__).parent / "web"

app = FastAPI()
if WEB_DIR.exists():
    app.mount("/web", StaticFiles(directory=WEB_DIR), name="web")

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9

def top_p_sample(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    top_p = max(1e-5, min(float(top_p), 1.0))
    temperature = max(float(temperature), 1e-5)

    probs = torch.softmax(logits / temperature, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
    cum_probs = torch.cumsum(sorted_probs, dim=-1)

    mask = cum_probs - sorted_probs > top_p
    sorted_probs[mask] = 0.0
    norm = sorted_probs.sum(dim=-1, keepdim=True)
    norm = torch.clamp(norm, min=1e-12)
    sorted_probs = sorted_probs / norm

    sample = torch.multinomial(sorted_probs, num_samples=1)
    return sorted_idx.gather(dim=-1, index=sample)

def generate(
    prompt: str,
    max_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    special_tokens: list[str] | None = None,
) -> str:
    special_tokens = special_tokens or SPECIAL_TOKENS

    token_ids = tokenizer.encode(prompt)
    tokens = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)

    limit = max(1, min(int(max_tokens), args["context_length"]))
    if tokens.size(1) >= limit:
        return ""
    print(limit, temperature, top_p)
    response: list[str] = []
    with torch.no_grad():
        while tokens.size(1) < limit:
            logits = model(tokens)[:, -1, :]
            next_token = top_p_sample(logits, temperature, top_p)
            tokens = torch.cat((tokens, next_token), dim=-1)

            next_word = tokenizer.decode([next_token.item()])
            if next_word in special_tokens:
                break

            response.append(next_word)

    return "".join(response)

@app.post("/generate")
def generate_endpoint(req: GenerateRequest):
    try:
        completion = generate(
            prompt=req.prompt,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
        )
        return {"completion": completion}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@app.get("/", response_class=HTMLResponse)
def index():
    if not WEB_DIR.exists():
        raise HTTPException(status_code=404, detail="Chat UI not found")
    index_path = WEB_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Chat UI not found")
    return index_path.read_text(encoding="utf-8")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)