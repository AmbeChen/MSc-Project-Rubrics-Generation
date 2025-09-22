# MSc-Project-Rubrics-Generation

This repository contains the code and data for an MSc project on **automatic rubric generation for medical dialogue evaluation**.  
The system provides both a **baseline RAG-based method** and an **extended multi-agent RAG pipeline**, together with evaluation scripts.

And the folder: LLMEval-Med contains the similar pipeline (a baseline and a multi-agent RAG-based). 

---

## 📌 Project Overview

The goal of this project is to automatically generate rubrics that can be used to evaluate medical dialogue systems.  
The pipeline combines **retrieval-augmented generation (RAG)** with few-shot prompting, and extends it with a **multi-agent framework** where different agents collaborate to improve rubric quality.

---

## 🔧 Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/<your-username>/MSc-Project-Rubrics-Generation.git
cd MSc-Project-Rubrics-Generation
pip install -r requirements.txt

## How to Run Everything 

> **Prereqs**
> - Install deps: `pip install -r requirements.txt`
> - Create an output folder: `mkdir -p outputs/baseline outputs/multi_agent outputs/eval`
> - Set API/model credentials if needed:
>   ```bash
>   # Serper API (for Mayo search)
>   export SERPER_API_KEY="YOUR_SERPER_API_KEY"
>   # If your HF model is gated/private:
>   export HUGGINGFACEHUB_API_TOKEN="YOUR_HF_TOKEN"
>   ```
> - Required input files (already in `data/`):
>   - `data/conversations_all.txt`
>   - `data/few_shot.jsonl`
>   - `data/reference_rubrics_all.jsonl`
```
---

### A) Data Preprocessing

Interactive (Jupyter):
1. Open `data_filter.ipynb`
2. Run all cells to produce filtered subsets if you need them (by default it reads/writes inside `data/`).

Headless (no UI):
```bash
jupyter nbconvert --to notebook --execute data_filter.ipynb --inplace
```
### B) Baseline (Vanilla RAG)

**Step B1 – Retrieve evidence from Mayo (via Serper)**

**Interactive (Jupyter):**
1. Open `baseline/mayo_retriever.ipynb`
2. Run all cells.

**Headless (no UI):**
```bash
jupyter nbconvert --to notebook --execute baseline/mayo_retriever.ipynb --inplace
```
### Step B2 – Generate rubrics (vanilla)

Interactive (Jupyter):

Open baseline/generate_rubrics.ipynb

Run all cells.

Headless (no UI):
```bash
jupyter nbconvert --to notebook --execute baseline/generate_rubrics.ipynb --inplace
```
