# OCR indexing for scanned ERP vouchers

The indexing half of a RAG system over scanned GP Petroleums vouchers. It turns folders of scanned PDFs into a searchable index where every hit cites an exact **file (voucher number) + page**. The query side and the chat front end are out of scope.

Each PDF is one Business Central voucher. Page 1 is the printed voucher with the document number, and the remaining pages are supporting papers: printed bills, handwritten claims, thermal slips, phone screenshots, faded receipts and some Hindi.

**No company data is in this repo.** PDFs, page images, run outputs (parquet/jsonl), the OCR answer key and the real question set are all excluded by `.gitignore`. The code finds the data on Kaggle as attached private datasets.

## What the pipeline does (`gp_indexing.py`, the same code as `gp_indexing.ipynb`)

| Step | What happens |
|---|---|
| Load | Finds every PDF under the input folder, keeps the `BE.../FE.../file.pdf` structure, and hashes each file (SHA-256). Only new or changed files are processed again. |
| CPU OCR and layout | RapidOCR and the scanner text layer give a quick first read, page flags (table, low confidence, handwriting, script check) and orientation. |
| Back-of-sheet pages | Detects stamp-only back sides where the front shows through as mirrored grey text, whites out the show-through and reads each coloured stamp as its own crop. |
| Page reader | **LightOnOCR-2-1B** reads every page. **Qwen3-VL** (8B if 20 GB+ is free, otherwise 4B) re-reads pages where LightOn returns empty, looping or degenerate output, and every page with Devanagari. If both fail, a de-looped read is kept. |
| Voucher number | Taken from page 1 and inherited by every page of the file, then cross-checked against the file name. Pages with no voucher number stay indexed as `UNASSIGNED` and are listed in `file_number_exceptions.csv`. |
| Chunking | The parent is the page and the child is a chunk of up to about 900 characters. Search matches children and returns each page once with its full text. |
| Metadata | `voucher_no`, `voucher_type`, `page_kind`, `ocr_conf`, `text_source`, and optional Business Central fields (`bc_vendor`, `bc_posting_date`, `bc_amount`, `bc_department`, `access_tags`) from a `bc_vouchers.csv` export. |
| Indexes | BM25 with a tokenizer that keeps IDs whole and compact (`IV/20/2223/0274`, `IV-20-2223-0274` and `iv2022230274` all match), plus dense **bge-m3** in HNSW, fused with RRF and reranked by **bge-reranker-v2-m3**. Access and metadata filters are applied before ranking. |
| Evaluation | Section 12b scores page hit@1/@5 against `eval_queries.csv` if one is attached. |

## Run on Kaggle

1. Upload the scans as a **private** Kaggle dataset (zip the folder that holds the `BE...` folders).
2. Code > New Notebook > File > Import Notebook > `gp_indexing.ipynb`.
3. Add Input: the scans dataset. Optional: a previous run's output as the cache, so unchanged pages are not read again; `eval_queries.csv`; `bc_vouchers.csv`.
4. Accelerator **GPU T4 x2**, Internet **On**, then Save & Run All (commit mode, because the run takes hours).
5. Outputs go to `/kaggle/working/index` (pages, chunks, indexes, `page_report.csv`, exceptions) and `/kaggle/working/cache`.

Keep the dataset and notebook private: outputs contain the OCR'd text. More detail, including the internet-off setup and the history of each fix, is in [`docs/kaggle_run_notes.md`](docs/kaggle_run_notes.md).

Question file format: see [`eval/eval_queries.example.csv`](eval/eval_queries.example.csv). `expected_pages` uses `|` between acceptable pages.

## Tests (no GPU, no models, no data)

```
pip install pandas numpy pillow scipy
python3 test_page_reader.py     # reader routing, de-looping, Hindi detection, show-through and stamp crops
python3 test_parent_child.py    # parent-child chunks, metadata filters, access tags
```

## OCR bake-off (`bakeoff/`)

`bakeoff.py` renders pages, runs each engine in its own virtualenv, and scores field recall against a typed answer key. `engines.py` holds one wrapper per engine, and `disagreements.py` lists values on which engines disagree, for a person to check. The answer key (`gt_fields.csv`) and the Kaggle notebooks that embed it are not in the repo. Full report: [`docs/OCR_comparison_2026-11-13.md`](docs/OCR_comparison_2026-11-13.md).

## Current results

Bake-off on 47 pages (4 vouchers), field recall against 377 typed values:

| Engine | Recall | Failed pages | T4 s/page |
|---|---|---|---|
| Claude Sonnet 5.5 (API, reference only) | 0.977 | 0% | 3.6 (API) |
| LightOnOCR-2 (chosen) | 0.934 | 8.5% | 32 |
| Qwen3-VL-8B | 0.914 | 2.1% | 23 |
| Qwen3-VL-4B | 0.907 | 2.1% | 21 |
| GLM-OCR | 0.845 | 0% | 13 |
| RapidOCR | 0.708 | 0% | 3.8 (CPU) |
| Scanner text layer | 0.663 | n/a | 0 |
| Tesseract | 0.549 | 2.1% | 8.5 (CPU) |

Caveats: the top local engines are tied within noise. The key was typed by Claude, which favours Claude. Precision (invented text) is not measured.

The full pipeline on the same 47 pages had an index-text recall of **0.930** (handwritten 0.84, poor scans 0.64).

On 313 pages (14 PDFs, run V4 on Nov 18), it ran in 4,320 s with earlier reads reused. A cold run takes about 5 h on T4 x2, with LightOn at about 53 s/page. Retrieval with hybrid + rerank scored a **page top-1 of 0.77 and top-5 of 0.95**, but only on 22 questions written by us after reading the pages. That is kinder than real user questions and is not an accuracy claim.

## Known gaps

- There is no real question set from the finance team yet, and 22 self-written questions are too few to rank retrieval options.
- Stamp digits on back-of-sheet pages are still unreliable. The stamp-crop read is built but has not been run on a GPU.
- Two truly sideways pages: the rotation result never reaches the VLM reader.
- Throughput: about 70-110 pages/hour on a T4 is fine for samples but not for the full archive. That needs faster GPUs or routing only flagged pages to the VLM.
- Storage hosting (Postgres/Qdrant) and the live Business Central link are not decided yet.
