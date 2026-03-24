# EcoLLM

EcoLLM is an AI-assisted life cycle assessment (LCA) project focused on extracting LCI information from documents, structuring expert decisions, and supporting downstream LCIA workflows.

## What this project does

- Parses and indexes LCA-related PDF documents
- Supports expert-in-the-loop data extraction with traceable action chains
- Provides API tools for scope definition, flow recording, parameter/calculation tracking, and summary retrieval
- Integrates flow matching (ecoinvent-style) and LCIA calculation workflow support
- Includes a fine-tuned LoRA model for tool-calling style LCA tasks

## Project structure

- `backend/`  
  FastAPI service layer and core logic: document processing, tool APIs, session management, LCI/LCIA services, matching, and chat endpoints.

- `frontend/`  
  Streamlit interface for document upload, AI chat, extraction workflow, flow matching, and LCIA operations.

- `scripts/`  
  Utility scripts for data preparation, export, evaluation, validation, and training pipeline support.

- `models/lca_lora/`  
  LoRA adapter artifacts (checkpoints, tokenizer/config files, training/eval logs) for the domain-tuned LCA assistant.
