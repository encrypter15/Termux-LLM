#!/data/data/com.termux/files/usr/bin/bash
set -e

echo "[1/7] Updating Termux packages..."
pkg update -y && pkg upgrade -y
pkg install -y python git cmake clang wget unzip

echo "[2/7] Installing Python dependencies..."
pip install --upgrade pip
pip install numpy hnswlib onnxruntime

echo "[3/7] Cloning and building llama.cpp..."
if [ ! -d llama.cpp ]; then
    git clone https://github.com/ggerganov/llama.cpp
fi
cd llama.cpp
make -j$(nproc)
cd ..

echo "[4/7] Downloading Phi-3 Mini quantized model..."
mkdir -p models
cd models
wget -c https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf
cd ..

echo "[5/7] Setting up Python project structure..."
mkdir -p termux_llm/{core,model,memory,connectors,personalization,api}

# core/config.py
cat > termux_llm/core/config.py <<'PY'
MODEL_PATH = "models/Phi-3-mini-4k-instruct-q4.gguf"
LLAMA_BIN = "llama.cpp/main"
EMBED_DIM = 384
PY

# model/engine.py
cat > termux_llm/model/engine.py <<'PY'
from abc import ABC, abstractmethod

class InferenceRequest:
    def __init__(self, prompt, max_tokens=256, temperature=0.7):
        self.prompt = prompt
        self.max_tokens = max_tokens
        self.temperature = temperature

class IEngine(ABC):
    @abstractmethod
    def load(self, model_path: str): ...
    @abstractmethod
    def generate(self, req: InferenceRequest) -> str: ...
    @abstractmethod
    def apply_lora(self, adapter_path: str): ...
PY

# model/llama_cpp_engine.py
cat > termux_llm/model/llama_cpp_engine.py <<'PY'
import subprocess
from .engine import IEngine, InferenceRequest

class LlamaCppEngine(IEngine):
    def __init__(self, binary_path):
        self.binary_path = binary_path
        self.model_path = None

    def load(self, model_path):
        self.model_path = model_path

    def generate(self, req: InferenceRequest) -> str:
        cmd = [
            self.binary_path,
            "-m", self.model_path,
            "-p", req.prompt,
            "--n-predict", str(req.max_tokens),
            "--temp", str(req.temperature)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.stdout

    def apply_lora(self, adapter_path):
        pass
PY

# memory/embeddings.py
cat > termux_llm/memory/embeddings.py <<'PY'
import onnxruntime as ort
import numpy as np

# Placeholder: replace with a real embedding model
def embed_text(text: str) -> np.ndarray:
    # For demo purposes, return a fixed-size random vector
    np.random.seed(abs(hash(text)) % (10**6))
    return np.random.rand(384).astype(np.float32)
PY

# memory/vector_store.py
cat > termux_llm/memory/vector_store.py <<'PY'
import hnswlib

class VectorStore:
    def __init__(self, dim, space='cosine'):
        self.index = hnswlib.Index(space=space, dim=dim)
        self.index.init_index(max_elements=10000, ef_construction=200, M=16)
        self.doc_map = {}

    def add(self, vectors, ids, docs):
        self.index.add_items(vectors, ids)
        for i, doc in zip(ids, docs):
            self.doc_map[i] = doc

    def search(self, vector, k=5):
        labels, distances = self.index.knn_query(vector, k=k)
        return [(self.doc_map[l], distances[0][i]) for i, l in enumerate(labels[0])]
PY

# api/cli.py
cat > termux_llm/api/cli.py <<'PY'
from termux_llm.model.llama_cpp_engine import LlamaCppEngine
from termux_llm.model.engine import InferenceRequest
from termux_llm.memory.vector_store import VectorStore
from termux_llm.memory.embeddings import embed_text
from termux_llm.core import config

def main():
    engine = LlamaCppEngine(config.LLAMA_BIN)
    engine.load(config.MODEL_PATH)

    vs = VectorStore(dim=config.EMBED_DIM)
    docs = [
        "Python is great for automation.",
        "Termux lets you run Linux on Android.",
        "llama.cpp enables running LLMs locally."
    ]
    embs = [embed_text(d) for d in docs]
    vs.add(embs, list(range(len(docs))), docs)

    query = input("Ask something: ")
    q_emb = embed_text(query)
    retrieved = vs.search(q_emb, k=2)
    context = "\n".join([doc for doc, _ in retrieved])

    prompt = f"Answer using this context:\n{context}\n\nQuestion: {query}"
    print(engine.generate(InferenceRequest(prompt)))

if __name__ == "__main__":
    main()
PY

echo "[6/7] Bootstrap complete."
echo "[7/7] Run the demo with:"
echo "python -m termux_llm.api.cli"
