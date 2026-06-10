# Embedder sweep (mid-size local models) — 6-query gold + abstention

Pipeline fixed (RRF + β=0 case-collapse + dense-gated pool); only the embedder varies. Local models 384-dim; NIM via Matryoshka truncation from 2048.

## A. Retrieval quality (core case recovered in top-5; rank shown)

| embedder | dim | SA 14-70905 | DV 23-4420 | POL 24-2787 | POL 20-72806 | vocab 25-120 | PSG 17-72197 | CITE 21-70493 | core rec |
|---|---|---|---|---|---|---|---|---|---|
| e5-small-v2 (current) | 384 | 4 | — | 1 | 4 | 1 | 1 | 5 | **6/7** |
| e5-base-v2 | 768 | — | 4 | — | 2 | 1 | 1 | — | **4/7** |
| e5-large-v2 | 1024 | — | 4 | 1 | 2 | 1 | 1 | — | **5/7** |
| gte-base | 768 | 2 | 4 | 1 | 5 | 2 | 1 | 4 | **7/7** |
| bge-base-en-v1.5 | 768 | — | 5 | 1 | 2 | — | 1 | — | **4/7** |
| nomic-embed-text-v1.5 | ? | ERROR: This modeling file requires the followin |||||||

## B. Abstention separation (dense max-cosine; gap = in-corpus min − max(absent,off))

| embedder | dim | in-corpus min | FGM/absent max | off-domain max | gap | separable? | params/int8/512MB |
|---|---|---|---|---|---|---|---|
| e5-small-v2 (current) | 384 | 0.824 | 0.819 | 0.779 | +0.005 | ~ thin | 33M / ~35MB / ✅ |
| e5-base-v2 | 768 | 0.771 | 0.789 | 0.745 | -0.018 | ✗ NO | 109M / ~110MB / ✅ |
| e5-large-v2 | 1024 | 0.781 | 0.793 | 0.726 | -0.012 | ✗ NO | 335M / ~335MB / ◑ tight |
| gte-base | 768 | 0.791 | 0.817 | 0.738 | -0.027 | ✗ NO | 109M / ~110MB / ✅ |
| bge-base-en-v1.5 | 768 | 0.538 | 0.574 | 0.447 | -0.036 | ✗ NO | 109M / ~110MB / ✅ |
