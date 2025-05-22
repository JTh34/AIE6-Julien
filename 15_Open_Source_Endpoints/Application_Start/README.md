---
title: Paul Graham Essay Bot
emoji: 📚
colorFrom: blue
colorTo: indigo
sdk: docker
sdk_version: "2.2.1"
app_file: app.py
pinned: false
---

# Paul Graham Essay Bot

This application uses Chainlit and LangChain to create a chatbot based on Paul Graham's essays. It uses Hugging Face endpoints for:

- Text embedding
- Text generation via an LLM

## Configuration

To run this application, you will need to set the following environment variables:
- `HF_LLM_ENDPOINT` - The Hugging Face endpoint URL for the LLM
- `HF_EMBED_ENDPOINT` - The Hugging Face endpoint URL for embeddings
- `HF_TOKEN` - Your Hugging Face API token

## Running