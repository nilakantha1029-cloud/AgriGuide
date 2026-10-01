# AgriGuide evaluation

33 answerable questions and 5 questions that should be refused, written from the indexed PDFs (expected source + page for each).

## Retrieval quality (higher is better)

| Method | Hit@1 | Hit@3 | Hit@5 | MRR |
|---|---|---|---|---|
| Vector only (FAISS) | 91% | 100% | 100% | 0.955 |
| Keyword only (BM25) | 73% | 91% | 97% | 0.823 |
| Hybrid (FAISS + BM25, RRF) | 94% | 97% | 100% | 0.961 |
| Hybrid + rerank | 88% | 97% | 100% | 0.927 |

Hit@k = the correct source page appears in the top k chunks. MRR = mean reciprocal rank of the first correct chunk.

## Refusal threshold

- Lowest top rerank score on an answerable question: **0.459** (q04: What moisture level should paddy have when stored?)
- Highest top rerank score on an out-of-domain question: **0.506** (u4: How do I train a dog to sit?)
- In-domain but missing from the documents (u1): **0.219** (a score threshold cannot catch this; the LLM `NOT_FOUND` check does)
- Suggested `MIN_RERANK_SCORE`: **0.05** (currently 0.03) - scores overlap, so the threshold only filters junk retrieval and the LLM `NOT_FOUND` check does the rest

## End-to-end answers

| Metric | Score |
|---|---|
| Answerable questions answered | 100% |
| Answers containing the expected key facts | 100% |
| Answers with source citations | 100% |
| Faithfulness (LLM judge: claims supported by context) | 94% |
| Correct refusals on unanswerable questions | 100% |
