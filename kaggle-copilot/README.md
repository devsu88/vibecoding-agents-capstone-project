# Kaggle Copilot Agent

An AI assistant that automatically decomposes a Kaggle competition description or URL into a functional, validated machine learning baseline solution. Built with Google Agent Development Kit (ADK) 2.0.

## Overview

The Kaggle Copilot accepts a Kaggle competition link, extracts context using a custom metadata scraping tool, analyses features, plans data preprocessing, designs machine learning pipelines, and evaluates candidates under target-leakage-protected splits. The workflow includes a **local sandboxed python runner** that tests the compiled pipeline and automatically requests self-correction loop revisions if compilation or runtime execution fails.

---

## 🏗️ Architecture

```mermaid
graph TD
    A[User Inputs URL] --> B[SSRF & URL Safe Check]
    B -->|Safe| C[Ingest Competition Metadata]
    C --> D[Identify Problem & EDA Plan]
    D --> E[Preprocessing Strategy]
    E --> F[Select Baseline Models]
    F --> G[Cross Validation & Metric Setup]
    G --> H[Combine & Generate Script]
    H --> I[verify_code_node Sandbox Execution]
    I -->|Error| J[Self-Correction Traceback Loop]
    J --> H
    I -->|Success| K[Ask for Review HITL]
    K -->|Revise| E
    K -->|Approve| L[Write baseline_solution.py]
```

### Key Workflow Features:
1.  **Streamlit Chat Interface**: A custom, polished UI featuring a ChatGPT-like sidebar for managing persistent conversation histories (`conversations.json`) across sessions.
2.  **Dynamic Graph Workflow**: Implemented as an asynchronous ADK dynamic flow, preserving loop checkpointing and variable scopes natively to support Human-in-the-loop (HITL) interactions.
3.  **Scraping Tool**: Ingests direct page descriptions via `fetch_kaggle_competition_metadata` to bypass guessing or prompt hallucinations.
4.  **Web Research**: Leverages `google_search` to actively research state-of-the-art model baselines tailored to the specific dataset structure and dynamically respects user-requested model constraints (e.g., forcing Neural Networks).
5.  **Sandbox Self-Correction Loop**: Validates syntax and executes mock fits in a subprocess before presenting deliverables, allowing the models to automatically resolve errors.

---

## 📁 Project Structure

```
kaggle-copilot/
├── app/                      # Core agent code
│   └── agent.py              # Main agent workflow logic & tools
├── streamlit_app.py          # Custom Streamlit Chat Frontend
├── conversations.json        # Persistent chat history storage
├── .env                      # Local environment configurations
├── pyproject.toml            # Project dependencies
└── README.md                 # Project guide
```

---

## ⚙️ Requirements & Installation

1. **uv**: Ensure Astral's Python manager `uv` is installed ([Install Guide](https://docs.astral.sh/uv/getting-started/installation/)).
2. **agents-cli**: Install via `uv tool install google-agents-cli`.
3. Configure the `.env` file at the root directory:
   ```env
   GOOGLE_CLOUD_PROJECT=your-gcp-project-id
   GOOGLE_CLOUD_LOCATION=global
   GOOGLE_GENAI_USE_VERTEXAI=True
   ```

Install project dependencies:
```bash
agents-cli install
```

---

## 🚀 Running the Agent

Start the local Streamlit application:
```bash
uv run streamlit run streamlit_app.py
```

1. Open the local web interface link shown in the terminal (usually `http://localhost:8501`).
2. Provide a Kaggle URL (e.g., `https://www.kaggle.com/competitions/titanic`) in the chat to begin.
3. The agent will fetch the metadata, plan preprocessing, research models, and sandbox the generated code.
4. When prompted by the agent, either reply with `approve` to finalize and write the script, or provide feedback (e.g., "Use XGBoost instead") to trigger a revision loop.
5. You can seamlessly switch between past projects using the **Past Conversations** menu in the sidebar!
