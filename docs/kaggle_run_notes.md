> Working notes copied from the Kaggle project folder. Files named here that hold company data (`ocr_ground_truth_starter.csv`, `gt_fields.csv`, `eval_queries.csv`, run outputs) are not in this repo. `gp_indexing_nov18.ipynb` is `gp_indexing.ipynb` here.

# Running the indexing notebook on Kaggle

## 0. Before you upload (do not skip)
This is company financial data (employee names, phone numbers, GSTINs, vehicle numbers, amounts).
- Get an OK from whoever owns data policy at GP Petroleums. "Private" on Kaggle still means stored on Google's servers.
- Dataset: **Private**. Notebook: **Private**. Never make either public: notebook outputs contain OCR'd text.
- The notebook uses only local open-source models. No data is sent to any API.

## 1. Upload the data as a private dataset
1. Zip the folder that contains the `BE...` folders (keep the `BE.../FE.../file.pdf` structure).
2. kaggle.com > Datasets > New Dataset > drop the zip. Title e.g. `gp-vouchers-sample`. Visibility: **Private**. Create.
   (CLI alternative: `kaggle datasets create -p <folder> --dir-mode zip`, which is private unless you pass `--public`.)
3. Kaggle unzips it to `/kaggle/input/gp-vouchers-sample/...`. The notebook finds every PDF under `/kaggle/input` itself.

## 2. Run
1. Kaggle > Code > New Notebook > File > Import Notebook > upload `gp_indexing.ipynb`.
2. Right panel: Add Input > your private dataset. Accelerator: GPU T4 x2 (needed for the LightOnOCR-2 reader, bge-m3 and the reranker;
   OCR itself runs on CPU). Internet: On (needs phone verification) for the pip/model downloads.
3. Run All. On the 47 sample pages the OCR step takes ~1.5 min on 4 CPU cores.
4. Optional: upload `ocr_ground_truth_starter.csv` (4 pages I transcribed) as a dataset to see section 8(c) numbers,
   then add your own transcriptions to the template the notebook writes to /kaggle/working.

## 3. Internet off
See the last cell of the notebook: build a private `gp-rag-wheels` dataset once (wheels + model snapshots), attach it.

## What was and was not tested
- Tested end to end on your 4 PDFs / 47 pages (CPU, Python 3.11): loading, OCR, layout, flags, file-number
  cross-check, accuracy check, chunking, BM25, HNSW, RRF, citations, saving.
- NOT tested: bge-m3, bge-reranker-v2-m3 and the section 6 page-reader cell (LightOnOCR-2 + Qwen3-VL-4B fallback; only its routing rules are unit-tested, `python3 test_page_reader.py`). Model downloads are blocked where this was built,
  so dense retrieval ran on a TF-IDF stand-in. First Kaggle run: check the cell 10 output says `dense: BAAI/bge-m3`.

## Parent-child chunks and metadata (added)
- **Parent = the page, child = a chunk of its text.** Most pages are one chunk, so parent and child are the same text. Pages longer than 900 characters are split into several children. Search matches children, returns each page once, and gives the full page as `parent_text`.
- **Metadata on every chunk:** `voucher_no`, `voucher_type`, `page_kind` (from OCR flags, a heuristic), `ocr_conf`, `bc_match`, `access_tags` and, if a Business Central export is provided, `bc_vendor`, `bc_posting_date`, `bc_amount`, `bc_department`.
- **Optional Business Central export:** add a private dataset containing `bc_vouchers.csv` (or set `GP_BC_CSV`) with columns `document_no, vendor, posting_date, amount, department, access_tags` (tags separated by `;`). Without it, BC columns are empty and every page has `access_tags = all`.
- **Filters and access:** `search(q, filters={"page_kind": ["handwritten_or_faded"]}, allowed_tags={"dept:LOG"})`. Pages the caller may not see are removed before ranking. Pages tagged `all` are visible to everyone.
- **Tested:** the new chunking, filter and parent-return logic passes a unit test on synthetic data (long page splits, access filter hides tagged pages, each page returned once). Not yet run on Kaggle with the real models: re-run the notebook and compare the 12b numbers with the previous run.
- **Run the check locally:** `python3 test_parent_child.py` (needs only pandas and numpy; no models or data).

## File cards (added 2026-11-12, not yet run on Kaggle with real models)
Section 10b builds one summary row per voucher (`cards.parquet`): header and Business Central fields, page kinds, amounts/dates/vehicle numbers from supporting pages, leading text of each page, and a `has_devanagari` flag. The same cell prints every page that contains Hindi. `search(..., use_cards=True)` matches the question against cards first and searches only the top 3 vouchers, falling back to all pages when no card matches. Section 12b prints the page hit@5 change with cards versus without, plus the questions that cards lost; adopt cards only if that change is about zero on a real question set. Card logic was unit-tested on a 3-page synthetic set with a stand-in BM25; the 22-question comparison has not been run.

## Page reader (Nov 13): LightOnOCR-2 primary, Qwen3-VL-4B fallback
- Section 6 now reads **every page** (`VLM_SCOPE = "all"`) with LightOnOCR-2-1B (fp32 on a T4). Bake-off field recall 0.934 vs RapidOCR 0.708 on the 47 sample pages.
- Pages where LightOn returns empty / looping / degenerate output (about 8% in the bake-off), and pages flagged as Hindi, are re-read by Qwen3-VL-4B. If both fail, RapidOCR text is used. `text_source` in the index says which reader produced each page.
- **Speed:** LightOn ~32 s/page median on a T4, about 110 pages/hour. RapidOCR was ~2,000 pages/hour. Fine for the sample; a full corpus needs a faster GPU, several GPUs, or `VLM_SCOPE = "flagged"`. `MAX_VLM_PAGES = 100` caps a run, so raise it for the 47-page sample plus new scans.
- **First run to do on Kaggle:** run the 47-page sample, check section 6 prints `primary: ... for 47 pages` and the fallback count, then compare section 8(c) with the bake-off numbers.

## Nov 16 fixes (reader routing + incremental re-indexing)
- **Why:** the Nov 13 Kaggle run (notebook "LightOnOCR-2") crashed the Qwen fallback on all 4 looping pages
  (`Invalid item number: i=np.int64(...)`), so they kept RapidOCR text, and the Hindi pages were never sent to Qwen.
- **Hindi routing:** a page goes to Qwen if LightOn's own output contains Devanagari (>= 10 chars) or section 4 flagged
  `check_script`. RapidOCR/scanner text cannot produce Devanagari, so the old rule could not see Hindi pages.
- **Fallback model:** Qwen3-VL-8B if the GPU has >= 20 GB free, else Qwen3-VL-4B (T4). Falls back to 4B if 8B fails to load.
- **Both readers loop:** the de-looped read is kept (`+deloop` in `text_source`), not RapidOCR.
- **Incremental:** OCR, reader and embedding caches are keyed by the SHA-256 of the PDF bytes. Section 3 prints
  new / changed / removed / unchanged files. Tested on CPU: touching 1 PDF and deleting 1 re-OCR'd only the touched
  file (10 s vs 74 s) and dropped the deleted one. Reader-cache second run read 0 pages (mock readers fed with the
  bake-off outputs). To reuse reads across Kaggle sessions, save `/kaggle/working/cache` as a private dataset and attach it.
- **Page report:** `index/page_report.csv` (reader, routing note, chars, Hindi flag per page).
- Early stopping of looping generation was tried and rejected: on bake-off outputs the stop rule also fired on
  real HTML tables (5 false stops), which would cut real text.

## Nov 16 Kaggle run, scored (Version 2, T4 x2, 45.5 min)
Field recall of the indexed page text vs the answer key (gt_fields.csv, typed by Claude): **0.930** (95% CI 0.89-0.96),
vs 0.934 for the bake-off LightOn reads; no measurable change. By type: printed 0.97, ERP voucher 0.98, screenshot 1.00,
thermal slip 0.97, handwritten 0.84, poor scan 0.64, Hindi 0.89. Per page: `run_nov16/page_scores.csv`.
Change after scoring: `UNION_ON_FALLBACK` (pages read by Qwen also index LightOn's de-looped read), est. 0.935.

## Nov 17: back-of-sheet stamp pages (show-through)
Run V3 (313 pages) showed 3 of 5 failed pages were not rotated but the back of a sheet: the front side shows through as faint
mirrored grey text and the real content is the purple inward stamps. LightOn looped on the mirrored text; Qwen invented dates
(1706 p32: wrote 25/04/22 and 25/05/12, page says 25/05/26; 1112 p4 index says 15 FEB 2026, stamp says 12 FEB 2026).
- `is_show_through_stamp_page` (dark < 0.2% and coloured-ink > 0.5% of pixels at 50 dpi) flags them. On the 266 new pages it
  flags exactly 12, all confirmed by eye on a contact sheet; 3 pages with similar "dark" share but real content are not flagged.
- `suppress_show_through` whites out grey pixels, keeps coloured ink and clearly dark ink. Checked by eye on 1706 p32 / p20: the
  mirrored invoice disappears, stamp numbers (0666, 25/05/26, 12:30, 01843) stay legible.
- Reader cache name gets an `ss` suffix for these pages, so only they (plus pages that never got cached) are re-read when
  V3's output is attached as input. `CACHE_VERSION` v4 (CPU OCR stage stores `stamp_only`).
- NOT yet verified through LightOn/Qwen: needs a Kaggle GPU run. Tests: `python3 test_page_reader.py`.
- The existing `ocr_with_orientation` is unchanged: it only tries 90/270 when the RapidOCR score is low and the result never
  reaches the VLM reader (`rotation` = 0 for all 313). Genuinely sideways pages (1059 p11, 1737 p7) are a separate open item.

## Nov 18: stamp-crop read (gp_indexing_nov18.ipynb)
V4 showed whole-page reads still garble stamp digits (1706 p32: Inward No 6666, date 25/05/22; true 0666, 25/05/26). Fix: on back-of-sheet pages, `stamp_boxes` finds each coloured-ink stamp, each crop is read alone by LightOnOCR, and the crop text REPLACES the whole-page text (note "stamp crops"; whole-page read stays in vlm_primary_raw). Cache suffix `ss2`, so only the 12 stamp pages re-read (about 2 min). Tests: test_page_reader.py. Not yet run on GPU.
