# OCR bake-off (Kaggle, T4 x2)

Upload `ocr_bakeoff.ipynb` to Kaggle (File > Import Notebook), attach the private `gp-petrochem-practice` dataset,
set Accelerator = GPU T4 x2 and Internet = On, then Run All. The notebook contains the code files itself
(`%%writefile` cells), so nothing else needs uploading.

Files:
- `ocr_bakeoff.ipynb`: the notebook (generated from the files below; edit those, not the notebook).
- `engines.py`: runs one engine on all pages in its own process. Output: `runs/<engine>.jsonl`, resumable.
- `bakeoff.py`: `render` / `setup` / `run` / `score`.
- `gt_fields.csv`: ground-truth sheet (not in this repo: it holds real values). `ocr_bakeoff.ipynb` and `claude_ocr_v2.ipynb` embed it, so they are left out too.

## Engines

| engine | role | licence | runs on |
|---|---|---|---|
| pdf_text | baseline (scanner's own text layer) | n/a | CPU |
| tesseract (eng+hin) | candidate | Apache-2.0 | CPU |
| rapidocr (current) | candidate | Apache-2.0 | CPU |
| ppocrv5_en, ppocrv5_hi | candidate | Apache-2.0 | CPU |
| paddleocr_vl (1.6) | candidate | Apache-2.0 | 1 T4 |
| glm_ocr | candidate | MIT | 1 T4 |
| lighton_ocr2 | candidate | Apache-2.0 | 1 T4 |
| qwen3vl_4b | candidate | Apache-2.0 | 1 T4 |
| qwen3vl_8b | candidate | Apache-2.0 | 2 T4 |
| chitrapathak2 | reference only | research-only | 1 T4 |
| chandra2 | reference only | free under $2M revenue | 1 T4 |

Surya OCR 2 is not included: its reader needs a vLLM or llama.cpp server, and vLLM on T4 is only proven via a custom build.

## What was tested and what wasn't

- **Tested here, end to end on all 47 real pages (CPU):** rendering and cropping, the pdf_text / Tesseract / RapidOCR engines,
  per-engine virtualenvs, the scheduler, resume, scoring, report and compare.html. The notebook itself was executed with those 3 engines.
- **Not tested:** every GPU model and PP-OCRv5. This build machine has no GPU and can't download models.
  Their code follows each model card's own transformers snippet, switched to fp16 and SDPA attention for T4.
  Cell 6 (smoke test, 2 pages each) exists to catch failures in minutes. Expect one or two engines to need a fix on first run;
  send me the tail of `logs/<engine>.log`.
- **Known handicaps:** PaddleOCR-VL and GLM-OCR are run on whole pages. Their official pipelines run a layout model
  first and read each block. Whole-page reading is what we'd use for routing, but it may understate them.
  If either is close to winning, a second round with their layout pipeline is worth doing.

## Scoring rules (bakeoff.py `field_hits`)

- amounts: digits compared after removing commas, `₹`, `/-` and trailing `.00`. An amount counts if the same number appears
  anywhere in the output, so a "30" elsewhere on the page also counts. This is lenient, so small amounts are easy to hit.
- dates: `14/10/22`, `14-10-2022`, `14 Oct 2022` and `October 14, 2022` match each other. If the ground truth has a year,
  the year must match too.
- ids: compared with spaces and punctuation removed, uppercase.
- names: share of the name's words found.
- hindi_words: exact Devanagari substring after Unicode NFKC normalisation.
- `ci95_low/high`: bootstrap over pages. Overlapping ranges mean the sample can't separate two engines.
- `agree_numbers`: agreement with the numbers most engines produced. This is not accuracy. It flags hard pages when there is no ground truth.
- `fail_rate`: error, empty output (<15 chars) or a repetition loop (output compresses to under 12% of its size).
