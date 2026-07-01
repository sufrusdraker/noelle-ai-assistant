# 🤖 NoellE - Multimodal & Interactive Local AI Ecosystem

**NoellE** is an artificial intelligence ecosystem designed to run 100% locally, optimized for CPU execution. The project evolved from a basic text-based conversational assistant into a modular multimodal platform, integrating Natural Language Processing (LLM), Image Generation via Diffusion (Stable Diffusion), real-time sentiment analysis with dynamic psychological states, and Computer Vision.

This repository documents both the current stable **production core** and the **research lab** containing advanced prototypes.

---

## 🚀 Key Features (Production - `src/`)

* **Real-Time Conversational Streaming:** Fluid, token-by-token text generation directly in the terminal using the **Llama-3-8B-Instruct** model via GPT4All.
* **Syntactic Command Orchestration:** A regex-based (`re`) parser monitors the LLM's output. When the AI decides to execute an action (e.g., `/image "prompt"`), the system intercepts the text and triggers the corresponding automation pipeline.
* **Instant Art Generation:** Integrated with the **SDXL Turbo** (Stable Diffusion) pipeline, configured for ultra-fast inference (only 4 steps) running locally on CPU.

---

## 🧪 Experimental Lab (`prototype/`)

The `prototype/` directory serves as the project's Research & Development (R&D) environment, housing complex modular implementations ready for future integration:

1. **`emotions.py` & `main_emotions.py` (Dynamic Emotional Engine):**
   * A psychological engine based on three cumulative numerical axes ranging from -10 to +10: **Connection** (Affinity), **Posture** (Defense/Pride Level - *Tsundere Style*), and **Stability** (Nervousness/Shyness).
   * It utilizes the `vaderSentiment` library combined with `deep-translator` to translate and quantify the emotional impact of user inputs in real time, dynamically updating the LLM's system prompt.
2. **`vision.py` (Visual Mapping - Image Captioning):**
   * Implementation of the Salesforce **BLIP** model via Hugging Face Transformers.
   * Enables the assistant to "see" local files, automatically converting image contents into highly detailed textual descriptions.
3. **`rag_pdf.py` (Local RAG Architecture - Retrieval-Augmented Generation):**
   * A lightweight search engine based on keyword matching that eliminates the need for heavy vector databases.
   * Extracts text from PDF files using `pypdf`, splits the content into chunks, and injects the exact context into the Llama-3 window to answer questions based on private local documents.
4. **`latent_callback.py` (VAE Latent Interception):**
   * An advanced mathematical experiment that injects a custom callback into the Stable Diffusion pipeline steps.
   * Intercepts latent tensors directly from the VAE at each inference step, denormalizes the data, and exports intermediate progress (`step_0.png`, `step_1.png`...), visualizing how the image is built out of pure noise.

### 🌐 API Development & HTTP Client-Server Architecture (`prototype/API_tests/`)
A dedicated microservices lab introducing server virtualization for the local intelligence:
* **Flask REST API:** Exposes the LLM as an HTTP service endpoint (`/chat`) accepting structured JSON payloads. Features regex-driven output post-processing to clean up raw generation tokens.
* **Persistent Session Memory:** Implements a localized state repository (`memoria_api.json`) that manages conversational history safely across server cycles, tracking client interactions to maintain dialog context.
* **Synchronous Test Client:** A specialized automation script using `requests` that acts as an isolated sandbox, testing endpoints, validating connection handshakes, handling timeouts, and parsing HTTP status responses.

---

## 🛠️ Tech Stack & Production Dependencies

The ecosystem relies on explicit version pinning to ensure mathematical and runtime stability across inference pipelines:

* **Core & Tensor Ops:** `torch==2.1.0`
* **Generative Diffusion & Vision:** `diffusers==0.24.0`, `transformers==4.35.2`, `accelerate==0.24.1`, `safetensors==0.4.1`, `Pillow==10.1.0`, `huggingface-hub==0.20.3`
* **LLM Core & RAG Integration:** `gpt4all==2.7.2`, `pypdf==4.2.0`
* **NLP & Affective Computing:** `vaderSentiment==3.3.2`, `deep-translator==1.11.4`, `numpy==1.26.4`
* **API Microservices:** `flask`, `requests`

---

## 📂 Repository Structure

```text
├── src/
│   ├── main.py               # Main stable assistant loop
│   └── pintora.py            # Stable image generation pipeline (SDXL Turbo)
├── prototype/
│   ├── emotions.py           # Emotional matrix calculation and axes algorithm
│   ├── main_emotions.py      # LLM integration with dynamic emotional state
│   ├── vision.py             # Image Captioning pipeline (Salesforce BLIP)
│   ├── rag_pdf.py            # Local document analysis engine (RAG)
│   ├── latent_callback.py    # Capturing intermediate latent states from diffusion
│   └── API_tests/
│       ├── server.py         # Flask API REST server exposing the LLM model
│       └── client.py         # Automated HTTP script to request endpoints
├── locais/
│   └── persona.txt           # Basic system personality guidelines
├── .gitignore                # Rules for tracking prevention (ignores virtual envs, models, and outputs)
├── requirements.txt          # Pinned version control of dependencies
└── README.md
