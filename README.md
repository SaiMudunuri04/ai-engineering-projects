# AI engineering projects

Seven independent, inspectable implementations spanning classical ML, deep learning, retrieval, multimodal search, agent orchestration, and parameter-efficient fine-tuning. Each folder below explains the problem, execution path, evaluation, and limits. These are **independent portfolio projects**, separate from employer work.

| Project | What is implemented | Evidence |
| --- | --- | --- |
| [Customer churn scoring](projects/customer-churn/README.md) | Chronological holdout, train-only standardization, regularized logistic model, calibration metric, model artifact | Executable CLI and unit tests |
| [Demand forecasting](projects/demand-forecast/README.md) | Lag/seasonal features, ridge regression, expanding-window backtest | Executable CLI and leakage test |
| [Visual defect classifier](projects/visual-defect-classifier/README.md) | PyTorch CNN, duplicate-image split check, early stopping, checkpoint and validation report | Training CLI; requires labelled images and optional vision dependencies |
| [Evidence RAG](projects/evidence-rag/README.md) | Document chunking, BM25 retrieval, grounded prompt, source IDs and citation validation | Retrieval and fake-generator tests; live generation needs an inference endpoint |
| [Multimodal search](projects/multimodal-search/README.md) | CLIP image/text embeddings, fusion, cosine retrieval, path validation | Ranking tests; live CLIP needs model download and image catalog |
| [Incident triage agent](projects/incident-agent/README.md) | Bounded plan-act-observe loop, read-only tool allowlist, audit trace | Tool-safety and step-limit tests; live planning needs an inference endpoint |
| [LoRA ticket classifier](projects/lora-text-classifier/README.md) | Binary text classification with LoRA, separated holdout, training loop, evaluation and adapter export | Data-leakage tests; full fine-tuning needs model download and optional dependencies |

## Quick start

Requires Python 3.10 or newer. The core projects and tests use only the standard library.

```bash
python3 -m pip install -e .
python3 scripts/create_demo_data.py
python3 -m portfolio_ai.core_ml data/churn_synthetic.csv
python3 -m portfolio_ai.forecasting data/demand_synthetic.csv
python3 -m unittest discover -s tests -v
```

The generated CSV and JSONL records are **synthetic smoke-test fixtures**, not business datasets. Reported metrics from these fixtures are not real-world performance claims. The vision, multimodal, and fine-tuning folders specify their optional packages and dataset layout.

## Delivery and credentials

GitHub Actions runs deterministic tests on Python 3.10 and 3.12. A `v*` tag builds a wheel and publishes it as a GitHub release after tests pass. No workflow deploys to AWS. `.env.example` lists placeholder settings; it contains no usable access keys. For a real AWS deployment, create an IAM role and use GitHub OIDC with least privilege, then add environment-specific deployment steps and validate them in your account. No production infrastructure or model results are claimed here.

The source is public for review. Model files, secrets, and local `.env` values are ignored. Do not commit private company documents, customer data, or credentials.
