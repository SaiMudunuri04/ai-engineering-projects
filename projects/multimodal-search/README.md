# Multimodal catalog search · image and text

Indexes an image catalog with CLIP visual embeddings and optional caption embeddings. A text query is embedded in the same space and ranked by cosine similarity. Catalog IDs must be unique, image paths must stay under the configured root, and embedding vectors are normalized.

Install with `python3 -m pip install -e '.[multimodal]'`. Prepare a JSONL catalog such as `{"id":"part-01","image_path":"part-01.png","caption":"metal bracket"}` plus the image files. Run `python3 -m portfolio_ai.multimodal catalog.jsonl images "metal bracket"`. The default checkpoint is `openai/clip-vit-base-patch32`; the first run downloads model weights.

No image collection or retrieval-quality score is bundled. To assess a real catalog, prepare labelled query–image pairs and report recall@k. The deterministic test uses a fake encoder to verify ranking behavior without claiming CLIP quality.
