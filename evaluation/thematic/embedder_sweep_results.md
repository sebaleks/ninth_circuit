# Embedder sweep — 6-query gold + abstention

Pipeline fixed (RRF + β=0 case-collapse + dense-gated pool); only the embedder varies. Local models 384-dim; NIM via Matryoshka truncation from 2048.

## A. Retrieval quality (core case recovered in top-5; rank shown)

| embedder | dim | SA 14-70905 | DV 23-4420 | POL 24-2787 | POL 20-72806 | vocab 25-120 | PSG 17-72197 | CITE 21-70493 | core rec |
|---|---|---|---|---|---|---|---|---|---|
| e5-small-v2 | 384 | 4 | — | 1 | 4 | 1 | 1 | 5 | **6/7** |
| bge-small-en-v1.5 | 384 | — | — | 1 | 2 | 3 | 1 | — | **4/7** |
| gte-small | 384 | — | — | 2 | 4 | 1 | 1 | — | **4/7** |
| all-MiniLM-L6-v2 | 384 | — | — | 1 | 3 | — | 1 | — | **3/7** |
| NIM nemotron @512 | 512 | — | 4 | 4 | 3 | 1 | 1 | 1 | **6/7** |
| NIM nemotron @1024 | 1024 | — | 4 | 3 | 4 | 1 | 1 | 1 | **6/7** |
| NIM nemotron @2048 | 2048 | — | 4 | 1 | 4 | 1 | 1 | 1 | **6/7** |

## B. Abstention separation (dense max-cosine; gap = in-corpus min − max(absent,off))

| embedder | dim | in-corpus min | FGM/absent max | off-domain max | gap | separable? |
|---|---|---|---|---|---|---|
| e5-small-v2 | 384 | 0.824 | 0.819 | 0.779 | +0.005 | ~ thin |
| bge-small-en-v1.5 | 384 | 0.646 | 0.625 | 0.477 | +0.021 | ✓ YES |
| gte-small | 384 | 0.826 | 0.815 | 0.751 | +0.011 | ~ thin |
| all-MiniLM-L6-v2 | 384 | 0.440 | 0.397 | 0.119 | +0.043 | ✓ YES |
| NIM nemotron @512 | 512 | 0.258 | 0.243 | 0.143 | +0.015 | ~ thin |
| NIM nemotron @1024 | 1024 | 0.245 | 0.177 | 0.121 | +0.069 | ✓ YES |
| NIM nemotron @2048 | 2048 | 0.255 | 0.167 | 0.095 | +0.088 | ✓ YES |
