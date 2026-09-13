# Evidence RAG · retrieval and grounded generation

Indexes local Markdown and text files into overlapping word chunks. A BM25 retriever selects passages; a chat generator receives a source-labelled prompt and must cite `[S1]`-style IDs. The result returns source text, rank scores, and a basic citation-ID check. Empty retrieval returns an explicit no-evidence answer.

Point it at an OpenAI-compatible **local or HTTPS** chat endpoint and run:

```bash
python3 -m portfolio_ai.rag data/documents "How should a worker be restarted?" \
  --model YOUR_LOCAL_MODEL --base-url http://127.0.0.1:8000/v1
```

Set `LLM_API_KEY` in your local environment only if the endpoint requires one. Do not index confidential documents into an untrusted service. Citation-ID validation checks formatting and source existence; it does not prove every factual claim is supported. Human review or entailment evaluation is needed before external use.
