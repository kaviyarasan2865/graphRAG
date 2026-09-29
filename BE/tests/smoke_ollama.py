"""Live smoke test against the local Ollama server. Not part of pytest suite."""

from app.providers.factory import get_embeddings, get_llm
from app.providers.base import ChatMessage


def main() -> None:
    emb = get_embeddings()
    vecs = emb.embed(["Olympic gold medal", "canoe sprint K-2 1000 metres"])
    print(f"embed provider={emb.name} model-dim(config)={emb.dim} actual-dim={len(vecs[0])} n={len(vecs)}")

    llm = get_llm()
    res = llm.chat(
        [
            ChatMessage("system", "Reply ONLY with compact JSON."),
            ChatMessage("user", 'Return {"ok": true, "sport": "canoeing"} as JSON.'),
        ],
        json_mode=True,
    )
    print(f"llm provider={llm.name} tokens={res.usage.as_dict()}")
    print("llm text:", res.text[:200])


if __name__ == "__main__":
    main()
