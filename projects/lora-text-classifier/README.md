# LoRA ticket classifier · parameter-efficient fine-tuning

Fine-tunes a binary support-ticket routing model with LoRA adapters on the DistilBERT attention query/value modules. Training and validation are separate JSONL files. Validation catches duplicate examples and cross-split text overlap. The script exports the adapter, tokenizer, and per-epoch accuracy, precision, recall, and F1 report.

Install with `python3 -m pip install -e '.[finetune]'`. Create fixtures with `python3 scripts/create_demo_data.py`, then run:

```bash
python3 -m portfolio_ai.fine_tuning \
  data/tickets_train_synthetic.jsonl data/tickets_val_synthetic.jsonl \
  --output artifacts/ticket-lora
```

The default checkpoint is `distilbert-base-uncased` and downloads on first use. The synthetic examples only verify the pipeline; their metrics do not demonstrate business performance. A credible application needs permissioned, labelled tickets, a later holdout, an unfine-tuned baseline, and error analysis before any deployment claim.
