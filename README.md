📱 Termux LLM – On‑Device AI with Personal Knowledge

Developer: Encrypter15 (Encrypter15@gmail.com)  
License: MIT  
Status: 🚧 In active development

---

📖 Overview

Termux LLM is a Python‑driven, modular, OOP‑architected large language model that runs entirely on an Android phone via Termux.  
It’s designed to be private, portable, and personalizable — capable of ingesting your own data, building a local knowledge base, and answering questions without sending anything to the cloud.

Key features:
- On‑device inference using llama.cpp with quantized models.
- Retrieval‑Augmented Generation (RAG) for personal knowledge integration.
- Modular OOP architecture with multiple Python packages for clean separation of concerns.
- LoRA adapter hot‑loading for lightweight personalization (trained off‑device, loaded on‑device).
- Extensible connectors for indexing local files, notes, and other data sources.

---

🏗 Architecture

`
termux_llm/
  core/              # Config, logging, events
  model/             # LLM engine interface + llama.cpp backend
  memory/            # Embeddings, vector store, retriever
  connectors/        # Data ingestion modules (files, notes, etc.)
  personalization/   # Profiles, prompt templates
  api/               # CLI and (future) HTTP API
`

Core Components
- Model Engine – Unified interface for multiple backends (llama.cpp, ONNX Runtime Mobile).
- Memory Layer – Embedding generation + vector search (HNSWlib).
- Connectors – Pluggable modules to feed the model with user‑approved data.
- Personalization – Profiles, prompt templates, and LoRA adapter management.

---

🚀 Quick Start

1. Install Termux & Dependencies
`bash
pkg update && pkg upgrade
pkg install python git cmake clang wget unzip
pip install numpy hnswlib onnxruntime
`

2. Clone & Build llama.cpp
`bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && make -j$(nproc) && cd ..
`

3. Download a Quantized Model
`bash
mkdir -p models
wget -O models/phi3-mini-q4.gguf \
  https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf
`

4. Run the Demo
`bash
python -m termux_llm.api.cli
`

---

🧠 How It “Learns”

This project doesn’t attempt full fine‑tuning on‑device (impractical for mobile hardware).  
Instead, it uses a three‑tier personalization strategy:

1. RAG Memory – Embeds and indexes your approved documents for context‑aware answers.
2. Profiles & Prompts – Adjusts tone, style, and domain focus.
3. LoRA Adapters – Optional lightweight fine‑tuning off‑device, then hot‑loaded on‑device.

---

📅 Roadmap

- [x] Termux bootstrap script
- [x] llama.cpp integration
- [x] Basic RAG pipeline
- [ ] Real ONNX embedding model (e.g., bge-small)
- [ ] File & note connectors
- [ ] LoRA adapter loader
- [ ] Local HTTP API for integrations
- [ ] Background indexing jobs

---

🤝 Contributing

Pull requests are welcome!  
If you have ideas for new connectors, optimizations, or model integrations, open an issue or PR.

---

📜 License

This project is licensed under the MIT License — see the LICENSE file for details.

---

📬 Contact

Developer: Encrypter15  
Email: Encrypter15@gmail.com  
GitHub: Encrypter15

