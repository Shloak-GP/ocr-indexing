# OCR engine comparison on GP Petroleums sample (13 Nov 2026, Claude row added 15 Nov)

**Test:** 12 local engines plus Claude Sonnet 5.5 through the API (15 Nov), on 47 scanned pages (4 vouchers). Score = field recall: the share of 377 typed values (amounts, dates, IDs, names, Hindi words) found in the engine's output. The typed values were read from the page images by Claude and spot-checked by the user. No engine output was used to write them. Speeds are seconds per page on a Kaggle T4 (ranking only; production GPUs are several times faster).

## Result in one line
A vision-language model (VLM) reads this data clearly better than RapidOCR, the scanner's text layer or Tesseract (about 0.91-0.93 recall against 0.55-0.71). The top four local VLM readers cannot be separated on 47 pages.

**Claude Sonnet 5.5 (API, 15 Nov) scored highest: 0.977, with no failures, at 3.6 s and $0.008 per page.** Treat that as "at least as good as the best local model", not "proven better": the answer key was typed by Claude, which favours Claude (see below).

## Overall (licence-clean engines first)
| Engine | Recall (95% range) | Failed pages | T4 s/page | Licence |
|---|---|---|---|---|
| Claude Sonnet 5.5 (API) | 0.977 (0.96-0.99) | 0% | 3.6 (API, not a T4) | paid API, $0.008/page; pages leave the company |
| LightOnOCR-2 | 0.934 (0.89-0.97) | 8.5% | 32 | Apache-2.0 |
| Qwen3-VL-8B | 0.914 (0.87-0.95) | 2.1% | 23 | Apache-2.0 |
| Qwen3-VL-4B | 0.907 (0.86-0.95) | 2.1% | 21 | Apache-2.0 |
| GLM-OCR | 0.845 (0.78-0.90) | 0% | 13 | MIT |
| PP-OCRv5 (en) | 0.783 | 0% | 41 (CPU) | Apache-2.0 |
| RapidOCR (current) | 0.708 (0.61-0.79) | 0% | 3.8 (CPU) | Apache-2.0 |
| Scanner text layer | 0.663 (0.54-0.76) | n/a | 0 | n/a |
| Tesseract | 0.549 (0.44-0.65) | 2.1% | 8.5 (CPU) | Apache-2.0 |
| Chandra-2 (reference) | 0.934, only 35 of 47 pages ran | 11.4% | 76 | restricted, free under $2M revenue |
| Chitrapathak-2 (reference) | 0.885 | 4.3% | 13 | research only |

## By page type (recall; n = pages)
Scored after the date-reader fix.

| Page type | n | Claude Sonnet 5.5 | LightOn | Qwen-8B | Qwen-4B | GLM-OCR | RapidOCR | Scanner text |
|---|---|---|---|---|---|---|---|---|
| Handwritten | 15 | 0.94 | 0.85 | 0.79 | 0.82 | 0.71 | 0.39 | 0.17 |
| Printed | 11 | 0.99 | 0.97 | 0.97 | 0.93 | 0.88 | 0.86 | 0.87 |
| Screenshot | 6 | 1.00 | 1.00 | 0.98 | 1.00 | 1.00 | 0.90 | 0.93 |
| Thermal slip | 6 | 0.97 | 0.93 | 0.87 | 0.88 | 0.88 | 0.74 | 0.62 |

Hindi (1 page) and poor scans (4 pages) are left out of this table because they are too small to rank engines; ERP vouchers (4 pages) score 0.79-0.98 for every engine and are omitted for space.

Handwriting is where the choice matters: the old text layer reads 17% and RapidOCR 39%, while the local VLMs read 71-85% and Claude 94% (range 0.88-0.98).

## Claude Sonnet 5.5 through the API (added 15 Nov)
Run on Kaggle, no GPU, one page at a time, model `claude-sonnet-5-5`, effort medium, page images capped at 1,568 px on the long side. Same pages, same answer key, same scorer as the rows above.

| Measure | Result |
|---|---|
| Field recall, all 47 pages | 0.977 (95% range 0.96-0.99) |
| Amounts / dates / IDs / names | 1.00 / 0.99 / 0.95 / 0.96 |
| Handwritten (15 pages, 85 values) | 0.94 (0.88-0.98); 7 pages missed one value each, mostly a handwritten year or a name |
| Poor scans (4 pages) | 0.93 |
| Hindi (IV-20-2223-0274 page 19) | 9 of 9 Devanagari words read, in the original script; the whole memo was transcribed, including the Marathi spellings |
| Failed pages | 0 of 47 (no loops, no refusals, no cut-off output) |
| Speed | 3.6 s median, 6.4 s at the 90th percentile, 12 s worst |
| Cost | $0.394 for 47 pages = $0.0084 per page (about 1,900 input and 460 output tokens per page). 100,000 pages: about $840, or about $420 with the Batch API (half price, results within 24 h) |

**How far to trust 0.977.** Lower than it looks, for three reasons.
1. **The answer key was typed by Claude** from the same page images. Where a handwritten digit is ambiguous, Claude typing the key and Claude reading the page tend to make the same call, so Claude gets credit and the other engines get marked wrong. Its lead over LightOn (0.977 against 0.934) is about the size this bias could produce, and the two 95% ranges almost touch. The handwriting gap (0.94 against 0.85) is the result most at risk, because that is where the key is least certain. The run is not just echoing the key: on IV-20-2223-0274 page 4 it read the year as "27" where the key says "22".
2. **Recall only.** The score counts values found and does not penalise invented or "corrected" text. Precision is unmeasured for every engine, Claude included.
3. **47 pages, one Hindi page.** Too few to rank Hindi reading at all.

To settle it, a person (not Claude) should check the values where Claude and LightOn disagree on the 15 handwritten pages; that is roughly 15-25 values and under an hour of work.

**Data privacy.** Every page image goes to Anthropic in the US. Anthropic's commercial API terms say API data is not used for training by default and is kept for a limited period (check the current terms before relying on this). Production use would need: company sign-off for sending financial documents (vendor names, GST numbers, employee names, amounts) outside, a check against India's data protection law (DPDP Act 2023) on cross-border transfer, and a key with spend limits that the company owns rather than a personal one. If that is not acceptable, the local engines above stay the plan, and Claude is still useful as a one-off checker to build a better answer key.

## Suggested routing (accuracy first, then speed; tied engines resolved by speed and licence)
If company policy allows sending scans to an outside API, Claude Sonnet 5.5 could replace the whole routing below: it was at least as accurate on every page type, about 6-9x faster per page than the T4 models, and has no GPU to run. If it does not allow that, use the local routing below.

- **Default reader for handwritten, poor scans and Hindi:** Qwen3-VL-4B (Apache). Within noise of LightOn and the 8B, runs at about 14 s per handwritten page, 0% failures on handwriting. LightOn is the alternative if accuracy is weighed above everything else, but it loops on about 1 page in 12.
- **Screenshots, ERP vouchers, thermal slips:** GLM-OCR (MIT, about 10-24 s, 0% failures, recall 0.88-1.00).
- **Cheap CPU fallback:** RapidOCR for printed pages only, with its 0.79 recall treated as a baseline, not a target.

## Not supported by this data
- The earlier routing table's winners are not reliable. It named PaddleOCR-VL best for ERP vouchers from one page at 928 s per page.
- Hindi is one page and 9 words. Chitrapathak-2 and LightOn read it, but one page cannot rank engines.
- Poor scans are 4 pages and 14 values; ranges are 0.5-1.0 wide.
- Top four engines are statistically tied overall. More pages (target 100+ hand-checked, including Hindi and poor scans) are needed before a final pick.
- A scorer bug was found and fixed: dates written like 22-Feb-22 were not recognised, so engines were marked wrong on one travel sheet (STF p2). The overall table is rescored with the fix; overall recall rose by about 0.02-0.03 for every engine and the ranking did not change.
- The raw results file lists PaddleOCR-VL first at 1.00, but it only ran on 2 pages at about 1,700 s each. Ignore that row.
- Chandra-2 ran on 35 of 47 pages and is licence-restricted. Chitrapathak-2 is research-only. Neither can be the production reader.
