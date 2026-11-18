# %% [markdown]
# # GP Petroleums scanned vouchers: indexing pipeline (Kaggle)
#
# **CONFIDENTIAL DATA.** These PDFs are company financial records with employee names, phone numbers,
# GSTINs and vehicle numbers. Before uploading anything to Kaggle:
# 1. Get written OK from whoever owns data policy at GP Petroleums. A *private* Kaggle dataset is still
#    stored on a third party's (Google's) servers. Private is necessary, not sufficient.
# 2. Keep the dataset **and this notebook** private. Notebook outputs (printed OCR text, search results)
#    are saved with every "Save Version". Making the notebook public publishes that text.
# 3. Everything here runs locally inside the Kaggle VM with open-source models. No hosted API is called.
#    Internet is used only to download packages/models (or not at all, see "Offline mode").
#
# **What this notebook does**
# 1. Finds every PDF under the dataset, records batch folder (`BE...`), file folder (`FE...`) and file name.
# 2. Per page: reads the scanner's embedded OCR text layer (Kodak Capture Pro), runs our own OCR (RapidOCR,
#    PP-OCRv4 ONNX models bundled in the pip wheel, so no download), runs layout detection (tables), and flags
#    pages that look handwritten / low-confidence.
# 3. Optional: sends only flagged pages (tables, handwriting) to a local vision-language model on the GPU.
# 4. Extracts the file number (voucher / document number) from page 1 and cross-checks it against the file name.
# 5. Measures OCR accuracy (free proxy metrics + a ground-truth template you fill in).
# 6. Parent-child chunks: the page is the parent (cited, returned in full), a chunk of its text is the child
#    (embedded, matched). Every chunk carries page metadata (voucher no., type, page kind, OCR confidence,
#    optional Business Central fields and access tags) that searches can filter on.
# 7. Builds BM25 + dense (HNSW) indexes, fuses with RRF, reranks with a cross-encoder.
# 8. Builds one file card per voucher (summary row used to rank and filter vouchers before page search) and
#    flags pages that contain Hindi (Devanagari).
# 9. Demo queries that return exact file + page citations.
#
# Re-runs are incremental: per-file OCR results are cached in `/kaggle/working/cache/`.

# %% [markdown]
# ## 1. Install
# Needs Internet ON (Settings > Internet; requires a phone-verified Kaggle account) **or** the offline wheel
# dataset described at the bottom. The data never leaves the VM either way.

# %%
import os, sys, subprocess
OFFLINE_WHEELS = "/kaggle/input/gp-rag-wheels"   # set if you built the offline bundle (see bottom)
PKGS = ["pymupdf", "rapidocr_onnxruntime", "rapid_layout", "bm25s", "faiss-cpu", "wordninja",
        "jiwer", "rapidfuzz", "sentence-transformers", "tqdm"]
src = ["--no-index", "--find-links", OFFLINE_WHEELS] if os.path.isdir(OFFLINE_WHEELS) else []
OCR_LIB = "/kaggle/working/pylibs"
if os.path.exists("/kaggle"):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + src + PKGS, check=False)   # deps
    # rapidocr_onnxruntime 1.4.4 declares Python <3.13, so on Kaggle's 3.13 pip falls back to 1.2.3 (string
    # scores, miscalibrated confidences). 1.4.4 runs fine on 3.13 (tested). Upgrading it in place left a
    # half-old/half-new package on Kaggle, so install 1.4.4 into its own folder and import it from there.
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--ignore-requires-python",
                    "--upgrade", "--target", OCR_LIB, "rapidocr_onnxruntime==1.4.4"] + src, check=False)
if os.path.isdir(OCR_LIB):
    sys.path.insert(0, OCR_LIB)
    for m in [m for m in sys.modules if m.startswith("rapidocr_onnxruntime")]:
        del sys.modules[m]                       # drop an old copy loaded earlier in this session
    import rapidocr_onnxruntime
    assert rapidocr_onnxruntime.__file__.startswith(OCR_LIB), rapidocr_onnxruntime.__file__
    print("rapidocr_onnxruntime loaded from", os.path.dirname(rapidocr_onnxruntime.__file__))

# %% [markdown]
# ## 2. Config

# %%
import glob, json, re, time, math, hashlib, random, unicodedata
from pathlib import Path
import numpy as np
import pandas as pd
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 20)

ON_KAGGLE = os.path.exists("/kaggle/input")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # less fragmentation; must be set before torch loads
# bm25s imports JAX if installed (Kaggle has it) and JAX grabs 75% of GPU 0 on first use. That was the
# 12 GB on gpu0 that PyTorch did not allocate. Keep JAX on CPU.
os.environ["JAX_PLATFORMS"] = "cpu"; os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
if "jax" in sys.modules:
    raise RuntimeError("JAX was loaded earlier in this session and is holding ~12 GB of GPU 0. "
                       "Run > Restart & clear cell outputs, then Run All.")
import shutil
HAS_GPU = shutil.which("nvidia-smi") is not None
# Folder that contains the BE.../FE.../*.pdf tree. Auto-detected on Kaggle; override if needed.
DATA_ROOT = os.environ.get("GP_DATA_ROOT", "/kaggle/input" if ON_KAGGLE else "./data")
WORK = Path(os.environ.get("GP_WORK", "/kaggle/working" if ON_KAGGLE else "./work"))
CACHE = WORK / "cache"; OUT = WORK / "index"
PREV_CACHE = os.environ.get("GP_CACHE_DIR") or (next(iter(glob.glob("/kaggle/input/**/cache/reader", recursive=True)), None) if ON_KAGGLE else None)
CACHE.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)

RENDER_DPI = 200            # scans are 200 dpi greyscale; rendering higher adds nothing
MAX_FILES = None            # e.g. 50 for a quick trial run
RUN_LAYOUT = True
CACHE_VERSION = "v4"        # bump to force re-OCR after changing section 4
# Incremental re-indexing: every cache is keyed by the SHA-256 of the PDF's bytes, not its path. A re-scanned file
# with the same name is re-read; a moved/renamed file is not; a deleted file drops out of the index. Point
# GP_CACHE_DIR at an attached dataset of an earlier run's /kaggle/working/cache to reuse its reads.

# Flag thresholds (tune after the accuracy check)
LOW_CONF = 0.80             # an OCR line below this confidence counts as "unsure"
FLAG_UNSURE_FRAC = 0.30     # page flagged if >30% of OCR lines are unsure
FLAG_MIN_CHARS = 40         # page flagged if OCR found almost nothing (blank, faded, photo)

# Page reader (GPU). Chosen on the 47-page bake-off (field recall 0.934 for LightOnOCR-2 vs 0.71 RapidOCR; see
# /mnt/project-files/ocr-bakeoff/OCR_comparison_2026-11-13.docx). LightOn fails ("loops" / empty output) on ~8% of
# pages, so those pages are re-read by Qwen3-VL-4B (2% failures, reads Hindi better). Hindi pages go to Qwen too.
RUN_VLM = HAS_GPU            # on by default when a GPU is attached; without a GPU pages fall back to RapidOCR
READER_PRIMARY = "lightonai/LightOnOCR-2-1B"        # Apache-2.0, ~1B, runs fp32 on a T4 (fp16 gives "!!!!")
# Fallback: Qwen3-VL-8B when the GPU has room (>= ~20 GB free, e.g. L4/A10/A100), else Qwen3-VL-4B (T4: 16 GB).
# Bake-off: 8B 0.914 vs 4B 0.907 overall, but on the Hindi page 8B found 3 of 4 Hindi fields, 4B 1 of 4.
READER_FALLBACKS = ["Qwen/Qwen3-VL-8B-Instruct", "Qwen/Qwen3-VL-4B-Instruct"]   # Apache-2.0. Avoid Qwen2.5-VL-3B.
FALLBACK_MIN_FREE_GB = {"Qwen/Qwen3-VL-8B-Instruct": 20, "Qwen/Qwen3-VL-4B-Instruct": 10}
READER_FALLBACK = READER_FALLBACKS[-1]   # replaced in section 6 by the largest one that fits
VLM_MODEL = READER_FALLBACK  # kept for older cells
UNION_ON_FALLBACK = True     # on pages read by the fallback, index LightOn's de-looped read too (recall over tidiness)
HINDI_TO_FALLBACK = True     # pages where LightOn wrote Devanagari (or flagged "check_script") are re-read by Qwen
MAX_VLM_PAGES = 300          # per run cap on pages NOT already in the reader cache; LightOn ~39 s/page on a T4
READER_MAX_NEW_TOKENS = 2048
# "all": every page (the bake-off scored page 1 too). "supporting": every page except page 1 (the clean ERP print).
# "flagged": only table / low-confidence pages. Cheaper; use at scale once the flag is shown to be reliable.
VLM_SCOPE = "all"

# Retrieval models (open source, run locally). If they cannot be loaded (no internet, no model dataset),
# the notebook falls back to a TF-IDF dense stand-in and skips reranking, and says so.
EMBED_MODEL = os.environ.get("GP_EMBED_MODEL", "BAAI/bge-m3")            # MIT, multilingual (Hindi too)
RERANK_MODEL = os.environ.get("GP_RERANK_MODEL", "BAAI/bge-reranker-v2-m3")  # Apache-2.0
CHUNK_CHARS, CHUNK_OVERLAP = 900, 150
# Optional Business Central export (CSV), joined to pages on the voucher number. Columns (any case):
#   document_no, vendor, posting_date, amount, department, access_tags   (access_tags: ';'-separated, e.g. dept:LOG;type:imprest)
# Without it the BC columns stay empty and every page is visible to everyone (access_tags = 'all').
BC_CSV = os.environ.get("GP_BC_CSV")
if not BC_CSV and ON_KAGGLE:
    BC_CSV = next(iter(glob.glob("/kaggle/input/**/bc_vouchers.csv", recursive=True)), None)

# %% [markdown]
# ## 3. Discover files

# %%
def find_pdfs(root):
    pdfs = sorted(p for p in Path(root).rglob("*") if p.suffix.lower() == ".pdf")
    return pdfs[:MAX_FILES] if MAX_FILES else pdfs

pdfs = find_pdfs(DATA_ROOT)
print(f"{len(pdfs)} PDFs under {DATA_ROOT}")

def folder_meta(p: Path):
    parts = p.parts
    be = next((x for x in parts if re.fullmatch(r"BE\d+", x)), None)   # batch folder
    fe = next((x for x in parts if re.fullmatch(r"FE\d+", x)), None)   # file folder
    return be, fe

import pymupdf as fitz  # PyMuPDF ("import fitz" is deprecated in PyMuPDF 2.x)
manifest = []
for p in pdfs:
    be, fe = folder_meta(p)
    with fitz.open(p) as d:
        manifest.append(dict(path=str(p), rel=str(p.relative_to(DATA_ROOT)), file_name=p.name,
                             file_hash=hashlib.sha256(p.read_bytes()).hexdigest(),
                             stem=p.stem, batch=be, folder=fe, n_pages=d.page_count,
                             producer=d.metadata.get("producer", "")))
manifest = pd.DataFrame(manifest)
print(manifest[["rel", "n_pages", "producer"]].to_string(max_rows=20))
print("total pages:", manifest.n_pages.sum())

# What changed since the last run (state saved at the end of section 13)
STATE_F = WORK / "index_state.json"
prev = json.loads(STATE_F.read_text()) if STATE_F.exists() else {}
now = dict(zip(manifest.rel, manifest.file_hash))
delta = dict(new=[r for r in now if r not in prev], changed=[r for r in now if r in prev and prev[r] != now[r]],
             removed=[r for r in prev if r not in now])
print({k: len(v) for k, v in delta.items()}, "| unchanged:", sum(prev.get(r) == h for r, h in now.items()))
for k in ("changed", "removed"):
    for r in delta[k][:20]: print(f"  {k}: {r}")

# %% [markdown]
# ## 4. Per-page extraction: text layer + OCR + layout + flags
# CPU speed with RapidOCR is roughly 0.5-2.5 s/page on Kaggle's 4 vCPUs (measured 0.5-2.4 s here on 4 cores).
# That is ~2,000-4,000 pages/hour. For the full corpus this notebook is a prototype, not the production run.

# %%
from rapidocr_onnxruntime import RapidOCR
ocr_engine = RapidOCR()
layout_engine = None
if RUN_LAYOUT:
    from rapid_layout import RapidLayout
    layout_engine = RapidLayout()   # PP-Layout CDLA (bundled): text/title/figure/table/header/footer...

def render(page, dpi=RENDER_DPI):
    pix = page.get_pixmap(dpi=dpi, colorspace=fitz.csRGB)
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)

# --- show-through core (unit-tested in test_page_reader.py) ---
def page_stats(arr):
    """(dark, stamp) fractions of a page image (uint8 RGB; ~50 dpi is enough).
    dark = share of clearly dark pixels (printed/handwritten ink); stamp = share of coloured pixels (purple ink stamps)."""
    a = arr.astype(np.int16); g = a.mean(axis=2); rel = g / max(float(np.percentile(g, 90)), 1.0)
    sat = a.max(axis=2) - a.min(axis=2)
    return float((rel < 0.55).mean()), float(((sat > 40) & (rel < 0.9)).mean())

def is_show_through_stamp_page(arr, max_dark=0.002, min_stamp=0.005):
    """Back of a sheet: the front side shows through as faint grey mirrored text, the real content is the ink
    stamps (inward no., dates). The readers loop on the mirrored text. Nov 17: 12 of 266 new pages, all checked by eye."""
    dark, stamp = page_stats(arr)
    return dark < max_dark and stamp > min_stamp

def suppress_show_through(arr, sat_min=22, dark_rel=0.5):
    """White-out grey (low-saturation) pixels; keep coloured ink and anything clearly dark. Removes the mirrored
    show-through, keeps stamps and handwriting."""
    a = arr.astype(np.int16); g = a.mean(axis=2); rel = g / max(float(np.percentile(g, 90)), 1.0)
    sat = a.max(axis=2) - a.min(axis=2)
    keep = ((sat >= sat_min) & (rel < 0.97)) | (rel < dark_rel)
    out = arr.copy(); out[~keep] = 255
    return out
def stamp_boxes(arr, sat_min=40, dil=25, min_area=0.004, pad=30):
    """Boxes (x0,y0,x1,y1) around clusters of coloured ink on a back-of-sheet page: each ink stamp is read on its own,
    because the whole-page read garbles stamp digits (Inward No 0666 -> 6666, 25/05/26 -> 25/05/22). Nov 18."""
    from scipy import ndimage as ndi
    a = arr.astype(np.int16); g = a.mean(axis=2); rel = g / max(float(np.percentile(g, 90)), 1.0)
    m = ndi.binary_dilation(((a.max(axis=2) - a.min(axis=2)) > sat_min) & (rel < 0.9), iterations=dil)
    lab, _ = ndi.label(m); H, W = m.shape; out = []
    for y, x in ndi.find_objects(lab):
        if (y.stop - y.start) * (x.stop - x.start) / (H * W) >= min_area:
            out.append((max(x.start - pad, 0), max(y.start - pad, 0), min(x.stop + pad, W), min(y.stop + pad, H)))
    return sorted(out, key=lambda b: (b[1], b[0]))
# --- end show-through core ---

def lines_from_ocr(res, h):
    """Group RapidOCR boxes into reading-order lines. Returns list of (y0, y1, text, conf)."""
    items = []
    for box, txt, conf in res or []:
        ys = [pt[1] for pt in box]; xs = [pt[0] for pt in box]
        items.append((min(ys), max(ys), min(xs), txt, float(conf)))
    items.sort(key=lambda t: ((t[0] + t[1]) / 2, t[2]))
    lines, cur = [], []
    for it in items:
        if cur and abs((it[0] + it[1]) / 2 - (cur[-1][0] + cur[-1][1]) / 2) > 0.5 * (it[1] - it[0]):
            lines.append(cur); cur = []
        cur.append(it)
    if cur: lines.append(cur)
    out = []
    for ln in lines:
        ln.sort(key=lambda t: t[2])
        out.append((min(t[0] for t in ln) / h, max(t[1] for t in ln) / h,
                    "  ".join(t[3] for t in ln), float(np.mean([t[4] for t in ln]))))
    return out

def run_ocr(img):
    """Returns [(box, text, conf_float)]. rapidocr_onnxruntime 1.2.x (what pip picks on Kaggle's Python 3.13)
    returns the confidence as a string; 1.4.x returns a float. Normalise both."""
    res, _ = ocr_engine(img)
    out = []
    for item in res or []:
        box, txt, conf = item[0], str(item[1]), item[2]
        try: conf = float(conf)
        except (TypeError, ValueError): conf = 0.0
        out.append((box, txt, conf))
    return out

def ocr_with_orientation(img):
    """Try 0 deg; if the page reads badly, try 90/270 (sideways scans) and keep the best."""
    best = (run_ocr(img), 0)
    def score(r): return sum(len(t) * c for _, t, c in r)
    if score(best[0]) < 200:
        for k, ang in ((1, 90), (3, 270)):
            r = run_ocr(np.ascontiguousarray(np.rot90(img, k)))
            if score(r) > 1.5 * score(best[0]):
                best = (r, ang)
    return best

def text_layer_quality(t):
    """Share of tokens that look like real words/numbers. Kodak's layer is good on printed text and
    junk like 'f^n* ^>17' on handwriting."""
    toks = t.split()
    if not toks: return 0.0
    good = sum(1 for w in toks if re.fullmatch(r"[A-Za-z]{2,}[.,:;]?|[\d.,/:-]+|[A-Z0-9/-]{3,}", w))
    return good / len(toks)

def extract_file(row):
    cache_f = CACHE / (hashlib.md5((CACHE_VERSION + row["file_hash"]).encode()).hexdigest() + ".json")
    if cache_f.exists():
        return json.loads(cache_f.read_text())
    pages = []
    with fitz.open(row["path"]) as d:
        for i, page in enumerate(d):
            t0 = time.time()
            layer = page.get_text("text")   # scanner OCR layer; order is messy, used for BM25 + comparison only
            img = render(page)
            h = img.shape[0]
            stamp_only = is_show_through_stamp_page(render(page, dpi=50))
            res, rot = ocr_with_orientation(img)
            lines = lines_from_ocr(res, h if rot == 0 else img.shape[1])
            ocr_text = "\n".join(l[2] for l in lines)
            confs = [c for _, _, c in res]
            unsure = float(np.mean([c < LOW_CONF for c in confs])) if confs else 1.0
            regions = []
            if layout_engine is not None:
                L = layout_engine(img)
                if L.boxes is not None:
                    regions = [dict(cls=c, score=round(float(s), 3), box=[round(float(v)) for v in b])
                               for b, c, s in zip(L.boxes, L.class_names, L.scores)]
            has_table = any(r["cls"] == "table" for r in regions)
            flags = []
            if has_table: flags.append("table")
            if unsure > FLAG_UNSURE_FRAC: flags.append("low_conf_or_handwriting")
            if len(ocr_text.replace(" ", "")) < FLAG_MIN_CHARS: flags.append("little_text")
            if re.search(r"[ऀ-ॿ]", ocr_text + layer) or (unsure > 0.5 and text_layer_quality(layer) < 0.4):
                flags.append("check_script")   # Devanagari or garbled: printed Hindi bills exist in the data
            header_text = "\n".join(l[2] for l in lines if l[0] < 0.30)   # top 30% of the page
            pages.append(dict(page=i + 1, text_layer=layer, ocr_text=ocr_text, header_text=header_text,
                              ocr_mean_conf=float(np.mean(confs)) if confs else 0.0, ocr_unsure_frac=unsure,
                              layer_quality=text_layer_quality(layer), rotation=rot, regions=regions,
                              page_flags=flags, stamp_only=stamp_only, secs=round(time.time() - t0, 2)))
    out = dict(rel=row["rel"], pages=pages)
    cache_f.write_text(json.dumps(out))
    return out

t_start = time.time()
extracted = []
for k, row in enumerate(manifest.to_dict("records")):
    extracted.append(extract_file(row))
    if k % 25 == 0:
        print(f"{k+1}/{len(manifest)} files, {time.time()-t_start:.0f}s")

page_rows = []
for row, ex in zip(manifest.to_dict("records"), extracted):
    for pg in ex["pages"]:
        page_rows.append({**{k: row[k] for k in ("rel", "file_name", "file_hash", "stem", "batch", "folder", "n_pages")}, **pg})
pages_df = pd.DataFrame(page_rows)
print(f"{len(pages_df)} pages in {time.time()-t_start:.0f}s")
print(pages_df.explode("page_flags")["page_flags"].value_counts(dropna=False))
print(f"{int(pages_df.stamp_only.sum())} back-of-sheet stamp pages (show-through suppressed before the reader sees them)")

# %% [markdown]
# ## 5. Voucher number per page: start pages split batch PDFs, cross-check with file name
# On this data the file number is the ERP voucher/document number printed on page 1
# (e.g. `Voucher No.: IV/20/2223/0274`, `Document No: NR/RPO/2223/1013`), and the PDF is named after it
# with `/` replaced by `-`. Supporting pages (receipts, bills) do **not** carry it reliably, so every chunk
# inherits the number from page 1.

# %%
ID_RE = re.compile(r"(?:voucher|document|doc)\s*no\.?\s*[:\-]?\s*([A-Z]{1,5}\s*[/\-]?\s*[A-Z0-9]{2,6}(?:\s*[/\-]\s*[A-Z0-9]{2,6}){1,3})", re.I)
# ^ the first "/" is optional: OCR drops it ("STF20/2627/0733"); norm_id puts it back.
GENERIC_RE = re.compile(r"\b([A-Z]{2,5}(?:/[A-Z0-9]{2,6}){2,4})\b")

def norm_id(s):
    if not s: return None
    s = re.sub(r"\s+", "", s.upper()).replace("-", "/")
    s = re.sub(r"^([A-Z]{1,5})(\d)", r"\1/\2", s)          # STF20/2627/0733 -> STF/20/2627/0733
    parts = s.split("/")
    # OCR confusions inside numeric segments
    fixed = [parts[0]] + [p.translate(str.maketrans("OILSB", "01158")) if sum(c.isdigit() for c in p) >= len(p) / 2 else p
                          for p in parts[1:]]
    return "/".join(fixed)

def find_file_number(*texts):
    for t in texts:
        if not t: continue
        m = ID_RE.search(t.replace("\n", " "))
        if m: return norm_id(m.group(1))
    for t in texts:
        if not t: continue
        m = GENERIC_RE.search(t)
        if m: return norm_id(m.group(1))
    return None

# Batch scans (Nov 16 email batch, HP scanner): one PDF holds SEVERAL vouchers, each starting with its printed ERP
# voucher page, and files are named "[Untitled].pdf". So the voucher number is assigned per PAGE: a page whose top
# 30% carries "Voucher No.: X" / "Document No: X" starts voucher X, and the following pages inherit it until the next
# start page. Pages before the first start page take the number from the file name if it is one, else "UNASSIGNED".
# The citation stays exact: file + PDF page; the voucher number is metadata on top.
def voucher_start(header_text, text_layer_head="", page=1):
    """'Voucher No.: X' anywhere in the page header starts voucher X. 'Document No: X' counts only on page 1: inside
    batch files, supplier tax invoices and e-way bills print 'Doc No.: Tax Invoice-...' and are NOT voucher starts."""
    for t in (header_text, text_layer_head):
        for m in ID_RE.finditer((t or "").replace("\n", " ")):
            kw = m.group(0).lower()
            if "voucher" in kw or page == 1:
                v = norm_id(m.group(1))
                if not v.startswith("TAX"):
                    return v
    return None

def looks_like_id(s):
    return bool(s) and bool(re.fullmatch(r"[A-Z]{1,5}(/[A-Z0-9]{2,6}){2,4}", s))

fn_rows, seg = [], {}
for rel, g in pages_df.sort_values("page").groupby("rel", sort=False):
    from_name = norm_id(g.stem.iat[0]) if looks_like_id(norm_id(g.stem.iat[0])) else None
    current, starts = from_name, []
    for idx, r in g.iterrows():
        v = voucher_start(r.header_text, r.text_layer[:600], int(r.page))
        if v:
            current = v; starts.append((int(r.page), v))
        seg[idx] = current or "UNASSIGNED"
    first = starts[0][1] if starts and starts[0][0] == 1 else None
    status = ("multi_voucher" if len(starts) > 1 else
              "no_voucher_page" if not starts else
              "match" if first == from_name else
              "no_filename_id" if from_name is None else "MISMATCH")
    fn_rows.append(dict(rel=rel, n_vouchers=len(starts), vouchers=";".join(f"p{p}:{v}" for p, v in starts),
                        from_filename=from_name, status=status,
                        unassigned_pages=int(sum(seg[i] == "UNASSIGNED" for i in g.index))))
fileno_df = pd.DataFrame(fn_rows)
print(fileno_df.status.value_counts().to_dict(), "| vouchers found:", int(fileno_df.n_vouchers.sum()))
print(fileno_df.drop(columns=["vouchers"]).to_string(max_rows=40))
# MISMATCH / no_voucher_page / unassigned pages go to a human: mis-named file, non-voucher document, or bad OCR.
fileno_df[(fileno_df.status.isin(["MISMATCH", "no_voucher_page"])) | (fileno_df.unassigned_pages > 0)].to_csv(
    OUT / "file_number_exceptions.csv", index=False)
pages_df["file_number"] = pd.Series(seg)
pages_df["doc_type"] = pages_df.file_number.str.split("/").str[0]

# %% [markdown]
# ## 6. Page reader (GPU): LightOnOCR-2, with Qwen3-VL (8B if it fits, else 4B) for failed and Hindi pages
# Bake-off result on 47 pages: LightOn field recall 0.934 (95% range 0.89-0.97), Qwen3-VL-8B 0.914, Qwen3-VL-4B 0.907,
# RapidOCR 0.708; the three VLMs cannot be separated on this sample. LightOn fails on 8.5% of pages (empty or
# looping output; Qwen 2.1%) and read the one Hindi page poorly, so:
# 1. pass 1: LightOnOCR-2 reads every page in scope;
# 2. pass 2: Qwen3-VL re-reads pages where LightOn failed (empty / looping / degenerate) and Hindi pages. A page is
#    Hindi if LightOn wrote Devanagari (it does: 194 and 122 Devanagari characters on IV/20/2223/0274 p15 and p19 in
#    the bake-off) or section 4 flagged `check_script`. RapidOCR and the scanner layer cannot see Devanagari at all;
#    on Nov 13 p15/p19 were never re-read because the old rule only looked at those two;
#    if both readers fail, the de-looped read is kept (looping reads start with the real page text);
# 3. a page where both fail keeps no VLM text, and section 7 falls back to RapidOCR for it.
# `vlm_reader` and `vlm_note` record which model produced each page, so a bad page can be traced.
# **First real run of this cell is yours**: this code has only been unit-tested on its routing rules, not run on a GPU.

# %%
import zlib
pages_df["vlm_text"] = None
pages_df["vlm_reader"] = None
pages_df["vlm_note"] = None
pages_df["vlm_primary_raw"] = None   # LightOn's own read, kept for pages where the fallback's text is used

# --- reader core (unit-tested in test_page_reader.py) ---
DEVANAGARI = re.compile(r"[ऀ-ॿ]")

def reader_failure(text):
    """'' if the output looks like a real page read, else why not. Same rule as the bake-off scorer."""
    t = (text or "").strip()
    if len(t) < 15:
        return "empty"
    if len(t) > 400 and len(zlib.compress(t.encode())) / len(t.encode()) < 0.12:
        return "loop"        # repeated output compresses far better than real page text
    if len(t) > 50 and max(t.count(c) for c in set(t)) / len(t) > 0.6:
        return "degenerate"  # e.g. "!!!!!" from fp16 overflow
    return ""

def has_hindi(text, min_chars=10):
    """True if a reader wrote Devanagari. The readers write Hindi in its own script, so this is the reliable
    Hindi detector; RapidOCR and the scanner text layer cannot produce Devanagari at all."""
    return len(DEVANAGARI.findall(text or "")) >= min_chars

def deloop(text, max_period=8, min_repeats=3):
    """Keep the real part of a looping read: a block of 1..max_period lines repeated min_repeats+ times in a row
    is kept once, then the same for a phrase repeated inside one line. Bake-off: looping pages usually read
    the top of the page correctly before they start repeating."""
    lines = (text or "").splitlines()
    out, i = [], 0
    while i < len(lines):
        for k in range(1, max_period + 1):
            blk = lines[i:i + k]
            if len(blk) < k or not any(l.strip() for l in blk):
                continue
            n = 1
            while lines[i + n * k:i + (n + 1) * k] == blk:
                n += 1
            if n >= min_repeats:
                out.extend(blk); i += n * k
                if i < len(lines) and blk[0].startswith(lines[i][:max(1, len(lines[i]))]) and i == len(lines) - 1:
                    i += 1                     # drop a cut-off last copy
                break
        else:
            out.append(lines[i]); i += 1
    t = "\n".join(out)
    return re.sub(r"(.{3,200}?)(?:\s*\1){%d,}" % (min_repeats - 1), r"\1", t, flags=re.S).strip()

def needs_fallback(text, flags, hindi_to_fallback=True):
    """Why a page must be re-read by the fallback model ('' = keep the primary read)."""
    why = reader_failure(text)
    if why:
        return "primary " + why
    if hindi_to_fallback and ("check_script" in (flags or []) or has_hindi(text)):
        return "hindi page"
    return ""

def pick_text(primary_text, fallback_text, why):
    """Returns (text, reader, note). reader is 'primary', 'fallback', 'primary_deloop' or 'fallback_deloop';
    text is None only if nothing usable is left (section 7 then uses RapidOCR)."""
    if not why:
        return primary_text, "primary", ""
    if not reader_failure(fallback_text):
        return fallback_text, "fallback", why
    if why == "hindi page":
        return primary_text, "primary", why + "; fallback failed, kept primary"
    # both failed: salvage the longer de-looped read
    cands = [(deloop(t), name) for t, name in ((primary_text, "primary_deloop"), (fallback_text, "fallback_deloop"))]
    cands = [(t, n) for t, n in cands if not reader_failure(t)]
    if cands:
        t, n = max(cands, key=lambda c: len(c[0]))
        return t, n, why + "; fallback failed too, kept de-looped text"
    return None, None, why + "; fallback failed too: " + (reader_failure(fallback_text) or "?")

# --- end reader core ---

if RUN_VLM:
    import torch, gc, transformers
    from PIL import Image
    for _name in ("embedder", "reranker", "vlm"):        # leftovers from an earlier run in this session
        if _name in globals(): del globals()[_name]
    gc.collect(); torch.cuda.empty_cache()
    free = [torch.cuda.mem_get_info(d)[0] / 1e9 for d in range(torch.cuda.device_count())]
    print("free GPU memory (GB):", [round(f, 1) for f in free])
    VLM_GPU = int(np.argmax(free))
    READER_FALLBACK = next((m for m in READER_FALLBACKS if free[VLM_GPU] >= FALLBACK_MIN_FREE_GB[m]), READER_FALLBACKS[-1])
    print("fallback reader:", READER_FALLBACK)
    if free[VLM_GPU] < 11:
        raise RuntimeError(f"Only {free[VLM_GPU]:.1f} GB free on the best GPU; the readers need up to ~11 GB. "
                           "Something from an earlier run still holds GPU memory. "
                           "Run > Restart & clear cell outputs, then Run All. (Or set RUN_VLM = False.)")

    OCR_PROMPT = ("Read all text in this image exactly as written, line by line, top to bottom. Include handwriting, "
                  "every number, date, amount and ID, and any Hindi (Devanagari) text in its original script. "
                  "Do not translate, summarise, correct or explain. Output plain text only.")

    def load_reader(kind):
        """kind 'primary' (LightOnOCR-2, fp32, no prompt) or 'fallback' (Qwen3-VL-4B, fp16). Returns (read, unload)."""
        if kind == "primary":
            repo, dtype, max_side = READER_PRIMARY, torch.float32, 1540
            cls = getattr(transformers, "LightOnOcrForConditionalGeneration")
            proc = getattr(transformers, "LightOnOcrProcessor").from_pretrained(repo)
            prompt = None
        else:
            repo, dtype, max_side = READER_FALLBACK, torch.float16, 1400
            cls = transformers.AutoModelForImageTextToText
            proc = transformers.AutoProcessor.from_pretrained(repo)
            prompt = OCR_PROMPT
        model = cls.from_pretrained(repo, dtype=dtype, attn_implementation="sdpa",
                                    device_map={"": f"cuda:{VLM_GPU}"}).eval()      # one GPU, not split

        def read(img):
            w, h = img.size
            sc = max_side / max(w, h)
            if sc < 1:
                img = img.resize((round(w * sc), round(h * sc)), Image.LANCZOS)
            content = [{"type": "image", "image": img}] + ([{"type": "text", "text": prompt}] if prompt else [])
            inputs = proc.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True,
                                              tokenize=True, return_dict=True, return_tensors="pt")
            inputs = {k: (v.to(model.device, dtype=dtype) if v.is_floating_point() else v.to(model.device))
                      for k, v in inputs.items()}
            with torch.inference_mode():
                out = model.generate(**inputs, max_new_tokens=READER_MAX_NEW_TOKENS, do_sample=False)
            return proc.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        def unload():
            nonlocal model
            del model; gc.collect(); torch.cuda.empty_cache()
        return read, unload

    if VLM_SCOPE == "all":
        todo = pages_df
    elif VLM_SCOPE == "supporting":
        todo = pages_df[pages_df.page > 1]
    else:
        todo = pages_df[pages_df["page_flags"].apply(lambda f: bool({"table", "low_conf_or_handwriting"} & set(f)))]
    # Reader cache: one JSON per page, keyed by file content hash + page + reader settings. Unchanged pages are
    # never re-read; changing a model, the prompt or the routing rules re-reads everything.
    READER_CFG = hashlib.md5(json.dumps([READER_PRIMARY, READER_FALLBACK, OCR_PROMPT, READER_MAX_NEW_TOKENS,
                                         HINDI_TO_FALLBACK, "route-v2"]).encode()).hexdigest()[:10]
    RCACHE = CACHE / "reader"; RCACHE.mkdir(exist_ok=True)
    def rcache_name(r): return f"{r.file_hash[:16]}_p{int(r.page)}_{READER_CFG}{'ss2' if r.stamp_only else ''}.json"
    def rcache_get(r):
        for d in [RCACHE] + ([Path(PREV_CACHE)] if PREV_CACHE else []):
            f = d / rcache_name(r)
            if f.exists(): (shutil.copy(f, RCACHE / f.name) if d != RCACHE else None); return json.loads(f.read_text())   # copy-forward: this run's output then holds every read, so the next run can reuse all of them
    cached = {idx: c for idx, r in todo.iterrows() if (c := rcache_get(r)) is not None}
    print(f"reader cache: {len(cached)} of {len(todo)} pages already read with these settings")
    todo_all, todo = todo, todo.drop(index=list(cached))
    if len(todo) > MAX_VLM_PAGES:
        print(f"!! {len(todo)} pages to read, capped at MAX_VLM_PAGES={MAX_VLM_PAGES}; the rest keep RapidOCR this run")
    todo = todo.head(MAX_VLM_PAGES)
    print(f"reading {len(todo)} pages with {READER_PRIMARY} on GPU {VLM_GPU}")
    paths = manifest.set_index("rel")["path"]

    def page_image(r):
        with fitz.open(paths.loc[r.rel]) as d:
            arr = render(d[int(r.page) - 1], dpi=200)   # int(): rows re-selected with .loc carry np.int64, which fitz rejects
            if r.stamp_only: arr = suppress_show_through(arr)
            return Image.fromarray(arr)

    def stamp_crops(r):
        """Separate crops of each ink stamp, rendered at 200 dpi from the unsuppressed page."""
        with fitz.open(paths.loc[r.rel]) as d: arr = render(d[int(r.page) - 1], dpi=200)
        return [Image.fromarray(arr[y0:y1, x0:x1]) for x0, y0, x1, y1 in stamp_boxes(arr)]

    def run_pass(rows, read, label):
        out, t_pass = {}, time.time()
        for idx, r in rows:
            t0 = time.time()
            try:
                out[idx] = read(page_image(r))
            except torch.cuda.OutOfMemoryError:
                gc.collect(); torch.cuda.empty_cache(); out[idx] = None
                print(f"  OOM on {r.file_number} p{r.page}")
            except Exception as e:                      # one bad page must not stop the run
                out[idx] = None; print(f"  error on {r.file_number} p{r.page}: {type(e).__name__}: {e}")
            print(f"[{label}] {r.file_number} p{r.page}: {time.time()-t0:.0f}s | {' '.join((out[idx] or '').split())[:100]}")
        print(f"{label}: {time.time()-t_pass:.0f}s for {len(rows)} pages")
        return out

    read1, unload1 = (load_reader("primary") if len(todo) else (None, lambda: None))
    primary = run_pass(list(todo.iterrows()), read1, "primary")
    stamp_txt = {}                                          # back-of-sheet pages: read each stamp on its own
    for idx, r in todo[todo.stamp_only.astype(bool)].iterrows():
        parts = []
        for k, crop in enumerate(stamp_crops(r), 1):
            try: t = deloop(read1(crop) or "")
            except Exception as e: t = ""; print(f"  stamp crop error {r.file_number} p{r.page} #{k}: {type(e).__name__}")
            if t.strip(): parts.append(f"[stamp {k}] {t.strip()}")
        if parts: stamp_txt[idx] = "\n".join(parts)
        print(f"[stamps] {r.file_number} p{r.page}: {len(parts)} stamps | {' '.join(stamp_txt.get(idx, '').split())[:140]}")
    unload1()

    why = {idx: needs_fallback(primary[idx], pages_df.at[idx, "page_flags"], HINDI_TO_FALLBACK) for idx in primary}
    redo = [(idx, pages_df.loc[idx]) for idx, w in why.items() if w]
    print(f"{len(redo)} of {len(todo)} pages need the fallback reader:",
          pd.Series([why[i] for i, _ in redo]).value_counts().to_dict() if redo else "none")
    fallback = {}
    if redo:
        try:
            read2, unload2 = load_reader("fallback")
        except Exception as e:                           # e.g. 8B does not fit after all: drop to the smallest
            print(f"!! could not load {READER_FALLBACK}: {type(e).__name__}: {str(e)[:150]}")
            gc.collect(); torch.cuda.empty_cache()
            READER_FALLBACK = READER_FALLBACKS[-1]; print("fallback reader:", READER_FALLBACK)
            read2, unload2 = load_reader("fallback")
        fallback = run_pass(redo, read2, "fallback")
        unload2()

    names = {"primary": READER_PRIMARY.split("/")[-1], "fallback": READER_FALLBACK.split("/")[-1]}
    for idx in primary:
        text, which, note = pick_text(primary[idx], fallback.get(idx), why[idx])
        if idx in stamp_txt:                                # stamp crops replace the whole-page read (which conflicts with them)
            text, note = stamp_txt[idx], "stamp crops"
        rec = dict(text=text, reader=(names[which.split("_")[0]] + ("+deloop" if "deloop" in which else "")) if which else None,
                   note=note, primary_raw=primary[idx], fallback_raw=fallback.get(idx))
        if primary[idx] is not None and (not why[idx] or fallback.get(idx) is not None):   # crashed reads are retried next run
            (RCACHE / rcache_name(pages_df.loc[idx])).write_text(json.dumps(rec, ensure_ascii=False))
        cached[idx] = rec
    for idx, rec in cached.items():
        pages_df.at[idx, "vlm_text"] = rec["text"]
        pages_df.at[idx, "vlm_reader"] = rec["reader"]
        pages_df.at[idx, "vlm_note"] = rec["note"]
        pages_df.at[idx, "vlm_primary_raw"] = rec.get("primary_raw")
    print(pages_df.loc[todo_all.index, "vlm_reader"].value_counts(dropna=False))
    pages_df[["rel", "page", "vlm_text", "vlm_reader", "vlm_note"]].to_parquet(OUT / "vlm_text.parquet")

# %% [markdown]
# ## 7. Pick the text to index for each page
# Measured on 4 hand-transcribed pages from this data (receipt, handwritten form, handwritten bill, app
# screenshot): RapidOCR found 56% of the true numbers vs 35% for the scanner's Kodak text layer, CER 0.19 vs 0.56.
# The Kodak layer skips handwriting and scrambles reading order; RapidOCR's weakness is gluing words
# (`GPPetroleums`), patched with `wordninja`. So: VLM text if present, else RapidOCR. The Kodak layer is
# still added to the BM25 text: it is free and sometimes reads a printed word RapidOCR missed.
# Re-check this rule once you have 30+ ground-truth pages (section 8); 4 pages is a hint, not proof.

# %%
import wordninja
def split_glued(text):
    """Re-insert spaces in long glued alphabetic runs from OCR ('OtherExpenses' -> 'Other Expenses')."""
    def fix(m):
        w = m.group(0)
        return " ".join(wordninja.split(w)) if len(w) >= 10 else w
    return re.sub(r"[A-Za-z]{10,}", fix, text)

def clean_vlm(t):
    """Qwen often wraps output in ```markdown fences and emits empty table skeletons
    ('| :--- | :--- |', '| | | |'). They carry no words but pad chunks and dilute embeddings."""
    t = re.sub(r"```[a-z]*", "", t)
    t = re.sub(r"\|([ \t]*:?-{2,}:?[ \t]*\|)+", "|", t)  # separator cells
    t = re.sub(r"(\|[ \t]*){2,}", "| ", t)                 # runs of empty cells
    lines = [l.strip(" |") for l in t.splitlines()]
    return "\n".join(l for l in lines if re.search(r"\w", l))

def choose_text(r):
    if isinstance(r.vlm_text, str) and r.vlm_text.strip():
        t = clean_vlm(r.vlm_text)
        # Fallback pages: also keep LightOn's de-looped read. Nov 16 run: Qwen-4B dropped a Hindi word on IV p19 that
        # LightOn had read before looping. Keeping both reads: est. field recall 0.935 vs 0.930 (estimate used the
        # bake-off LightOn reads for the 6 routed pages).
        prim = r.get("vlm_primary_raw")
        if UNION_ON_FALLBACK and "Qwen" in str(r.vlm_reader) and isinstance(prim, str) and prim.strip():
            extra = clean_vlm(deloop(prim))
            if extra and not reader_failure(extra):
                t = t + "\n" + extra
        return t, (r.vlm_reader or "vlm")
    if len(r.ocr_text.strip()) < 10 and r.layer_quality >= 0.6:   # OCR found nothing, layer has something
        return r.text_layer, "text_layer"
    return split_glued(r.ocr_text), "rapidocr"

pages_df[["index_text", "text_source"]] = pages_df.apply(lambda r: pd.Series(choose_text(r)), axis=1)
pages_df["bm25_text"] = (pages_df.text_layer + "\n" + pages_df.ocr_text.map(split_glued) + "\n"
                        + pages_df.vlm_text.fillna(""))
print(pages_df.text_source.value_counts())

# Page-level report: which reader produced each page and why. Saved to index/page_report.csv.
pages_df["has_hindi"] = pages_df.index_text.map(has_hindi)
page_report = pages_df[["rel", "page", "text_source", "vlm_note", "has_hindi"]].assign(
    chars=pages_df.index_text.str.len(), flags=pages_df.page_flags.map(",".join))
page_report.to_csv(OUT / "page_report.csv", index=False)
print(page_report[page_report.vlm_note.fillna("") != ""].to_string(index=False))
print("pages with Hindi text indexed:", page_report[page_report.has_hindi][["rel", "page"]].values.tolist())

# %% [markdown]
# ## 8. OCR accuracy check
# Three levels, cheapest first. Only (c) is a real accuracy number.
#
# **(a) File-number recovery rate** (free ground truth: the file name). Share of files where OCR read the
# voucher number correctly from page 1. Measures printed-header accuracy only.
#
# **(b) Engine agreement** between the scanner's text layer and RapidOCR (character error rate between them,
# spaces ignored). High disagreement = at least one engine is wrong. Not accuracy, but it ranks pages to review.
#
# **(c) Ground-truth CER/WER** on a stratified sample you transcribe by hand. The cell exports page images and
# a CSV; type the true text into `ground_truth`, upload the CSV as a dataset (or put it in /kaggle/working),
# re-run. 30-50 pages, spread across printed / handwriting / table / faded, is enough to make decisions.

# %%
import jiwer
from collections import Counter
def _norm(s): return unicodedata.normalize("NFKC", s or "").lower()
def cer(ref, hyp):
    """Order-sensitive character error rate, spaces ignored. Punishes reading-order differences too."""
    ref, hyp = re.sub(r"\s+", "", _norm(ref)), re.sub(r"\s+", "", _norm(hyp))
    if not ref: return float("nan")
    return min(jiwer.cer(ref, hyp), 1.0) if hyp else 1.0
def words(s): return [w for w in re.findall(r"[a-z0-9\u0900-\u097f]+", _norm(s)) if len(w) >= 2]
def word_recall(ref, hyp, numbers_only=False):
    """Share of the true words (or numbers) that appear anywhere in the OCR output. This is what retrieval
    needs: BM25 can only match words that were read. Order-insensitive."""
    ref_w = [w for w in words(ref) if (not numbers_only or any(c.isdigit() for c in w))]
    if not ref_w: return float("nan")
    hyp_bag = set(words(hyp)) | set(words(split_glued(hyp or "")))
    hyp_bag |= {re.sub(r"[^a-z0-9]", "", w) for w in _norm(hyp).split()}
    return sum(w in hyp_bag for w in ref_w) / len(ref_w)
def trigram_agreement(a, b):
    ta = Counter(re.sub(r"\s+", "", _norm(a))[i:i+3] for i in range(max(0, len(re.sub(r"\s+", "", a or "")) - 2)))
    tb = Counter(re.sub(r"\s+", "", _norm(b))[i:i+3] for i in range(max(0, len(re.sub(r"\s+", "", b or "")) - 2)))
    if not ta and not tb: return float("nan")
    return 2 * sum((ta & tb).values()) / (sum(ta.values()) + sum(tb.values()))

_named = fileno_df[fileno_df.from_filename.notna()]
print("(a) files named after a voucher whose page-1 number matches the name:",
      f"{(_named.status == 'match').mean():.0%} of {len(_named)}" if len(_named) else "n/a",
      f"| batch files: {int((fileno_df.status == 'multi_voucher').sum())},",
      f"{int(fileno_df.n_vouchers.sum())} vouchers found, {int(fileno_df.unassigned_pages.sum())} pages without a voucher")

pages_df["stratum"] = pages_df["page_flags"].map(lambda f: f[0] if f else "clean")
pages_df["layer_ocr_agreement"] = [trigram_agreement(a, b) for a, b in zip(pages_df.text_layer, pages_df.ocr_text)]
print("(b) agreement between scanner text layer and RapidOCR (char-trigram overlap, 1 = identical):")
print(pages_df.groupby("stratum").layer_ocr_agreement.describe()[["count", "mean", "min"]].round(2))
print("    lowest-agreement pages (review these first):")
print(pages_df.nsmallest(8, "layer_ocr_agreement")[["file_number", "page", "stratum", "layer_ocr_agreement"]].to_string(index=False))

GT_CSV = WORK / "ocr_ground_truth.csv"
GT_IN = next(iter(glob.glob("/kaggle/input/**/ocr_ground_truth.csv", recursive=True)), None) if ON_KAGGLE else None
gt_path = Path(GT_IN) if GT_IN else GT_CSV
if gt_path.exists() and pd.read_csv(gt_path).ground_truth.notna().any():
    gt = pd.read_csv(gt_path).dropna(subset=["ground_truth"])
    gt["file_name"] = gt.rel.map(lambda x: Path(x).name)   # match by file name: dataset root differs per upload
    gt = gt.drop(columns=["rel"]).merge(pages_df.drop(columns=["stratum"]), on=["file_name", "page"])
    rows = []
    for _, r in gt.iterrows():
        for eng, hyp in (("text_layer", r.text_layer), ("rapidocr", r.ocr_text), ("chosen", r.index_text),
                         ("vlm", r.vlm_text if isinstance(r.vlm_text, str) else None)):
            if hyp is None: continue
            rows.append(dict(stratum=r.stratum, engine=eng, word_recall=word_recall(r.ground_truth, hyp),
                             number_recall=word_recall(r.ground_truth, hyp, numbers_only=True),
                             cer=cer(r.ground_truth, hyp)))
    acc = pd.DataFrame(rows)
    print(f"(c) ground truth on {len(gt)} pages. word/number recall: higher is better; CER: lower is better")
    print(acc.groupby(["engine"])[["word_recall", "number_recall", "cer"]].mean().round(3))
    print(acc.groupby(["stratum", "engine"])[["word_recall", "number_recall"]].mean().round(2).unstack("engine"))
else:
    sample = pd.concat([g.sample(min(len(g), 10), random_state=0) for _, g in pages_df.groupby("stratum")])
    (WORK / "gt_pages").mkdir(exist_ok=True)
    for _, r in sample.iterrows():
        with fitz.open(manifest.set_index("rel").loc[r.rel, "path"]) as d:
            d[r.page - 1].get_pixmap(dpi=110).save(WORK / "gt_pages" / f"{r.stem}_p{r.page}.png")
    sample[["rel", "page", "stratum"]].assign(image=sample.stem + "_p" + sample.page.astype(str) + ".png",
                                              ground_truth="").to_csv(GT_CSV, index=False)
    print(f"(c) no ground truth yet. Wrote a {len(sample)}-page template to {GT_CSV} and images to "
          f"{WORK/'gt_pages'}. Type each page's text into ground_truth, upload the CSV, re-run this cell.")

# %% [markdown]
# ## 9. Parent-child chunks and page metadata
# **Parent = the page**: what a user cites and what a reader needs in full. **Child = a chunk of that page's
# text**: what gets embedded and matched. Most pages here are short (receipts, one form), so most pages have
# exactly one child and parent and child are the same text. Only long pages are split into several children.
# A child never crosses a page. Search matches children and returns each page once, with its full text.
#
# Every child carries the page's metadata, so searches can filter before ranking: voucher number and type, page
# kind (from the OCR flags: a heuristic, it misses some handwritten bills), OCR confidence, optional Business
# Central fields (vendor, date, amount, department) and access tags. Each chunk also gets a one-line context
# header (file number, document type, page x/y, vendor and date when known) prepended **for embedding only**.
# Without it, a toll receipt on page 13 has no idea which voucher it belongs to.

# %%
def split_page(text, size=CHUNK_CHARS, overlap=CHUNK_OVERLAP):
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text).strip()
    if len(text) <= size: return [text] if text else []
    chunks, start = [], 0
    while start < len(text):
        end = min(len(text), start + size)
        if end < len(text):                           # prefer to cut at a line break
            cut = text.rfind("\n", start + size // 2, end)
            end = cut if cut > 0 else end
        chunks.append(text[start:end].strip())
        if end >= len(text): break
        start = max(end - overlap, start + 1)
    return [c for c in chunks if c]

DOC_TYPES = {"IV": "Imprest voucher", "STF": "Employee/staff voucher", "TPT": "Transport voucher",
             "NR": "Credit note request"}   # extend as you see more prefixes

def page_kind(r):
    """Coarse page type from OCR flags. Page 1 is the printed ERP voucher. Heuristic, not verified."""
    f = set(r["page_flags"])
    if r.page == 1: return "erp_voucher"
    if "low_conf_or_handwriting" in f: return "handwritten_or_faded"
    if "table" in f: return "printed_table"
    return "printed"

def load_bc(path):
    """Business Central export keyed by normalised voucher number, or None if not provided."""
    if not path or not os.path.exists(path): return None
    bc = pd.read_csv(path)
    bc.columns = [c.strip().lower() for c in bc.columns]
    assert "document_no" in bc.columns, "BC export needs a document_no column"
    for c in ("vendor", "posting_date", "amount", "department", "access_tags"):
        if c not in bc.columns: bc[c] = None
    bc["voucher_key"] = bc.document_no.astype(str).map(norm_id)
    return bc.drop_duplicates("voucher_key").set_index("voucher_key")

bc = load_bc(BC_CSV)
print("Business Central export:", f"{len(bc)} vouchers from {BC_CSV}" if bc is not None
      else "not provided: BC columns stay empty, access_tags = 'all'")

BC_COLS = {"vendor": "bc_vendor", "posting_date": "bc_posting_date", "amount": "bc_amount", "department": "bc_department"}
pages_df["voucher_no"] = pages_df.file_number
pages_df["voucher_type"] = pages_df.doc_type
pages_df["page_kind"] = pages_df.apply(page_kind, axis=1)
pages_df["ocr_conf"] = pages_df.ocr_mean_conf.round(3)
pages_df["parent_id"] = pages_df.file_hash.str[:12] + "#p" + pages_df.page.astype(int).astype(str)   # unique per file page
for src_col, dst_col in BC_COLS.items():
    pages_df[dst_col] = pages_df.voucher_no.map(bc[src_col]) if bc is not None else None
if bc is not None:
    pages_df["bc_match"] = np.where(pages_df.voucher_no.isin(bc.index), "matched", "not_in_bc")
    pages_df["access_tags"] = pages_df.voucher_no.map(bc.access_tags).fillna("all").replace("", "all")
else:
    pages_df["bc_match"] = "no_bc_data"
    pages_df["access_tags"] = "all"

META_COLS = ["voucher_no", "voucher_type", "page_kind", "ocr_conf", "bc_match", "access_tags", *BC_COLS.values()]

def header(r):
    h = (f"File {r.file_number} | {DOC_TYPES.get(r.doc_type, r.doc_type)} | "
         f"page {r.page} of {r.n_pages} | {r.file_name}" + (f" | folder {r.batch}/{r.folder}" if r.batch else ""))
    if pd.notna(r.bc_vendor): h += f" | vendor {r.bc_vendor}"
    if pd.notna(r.bc_posting_date): h += f" | date {r.bc_posting_date}"
    return h

chunk_rows = []
for _, r in pages_df.iterrows():
    kids = split_page(r.index_text) or [""]
    hdr = header(r)
    for j, c in enumerate(kids):
        chunk_rows.append(dict(chunk_id=f"{r.parent_id}#c{j}", parent_id=r.parent_id, child_idx=j, n_children=len(kids),
                               file_number=r.file_number, file_name=r.file_name, rel=r.rel, batch=r.batch,
                               folder=r.folder, page=int(r.page), n_pages=int(r.n_pages), text=c,
                               embed_text=hdr + "\n" + c, bm25_text=hdr + "\n" + (r.bm25_text if j == 0 else c),
                               text_source=r.text_source, page_flags=",".join(r["page_flags"]),
                               **{m: r[m] for m in META_COLS}))
chunks = pd.DataFrame(chunk_rows)
print(len(chunks), "chunks;", chunks.text.str.len().describe()[["mean", "max"]].round(0).to_dict())

# Parents: one row per page with the FULL page text (what a reader gets once a child matches).
parents = (pages_df.assign(text=pages_df.index_text)
           [["parent_id", "file_name", "rel", "batch", "folder", "page", "n_pages", "text_source", "text", *META_COLS]]
           .set_index("parent_id", drop=False))
parents["n_children"] = chunks.groupby("parent_id").size()
print(f"{len(parents)} parents (pages); pages split into several children:", int((parents.n_children > 1).sum()))
print(parents.page_kind.value_counts().to_dict())

def visible_mask(filters=None, allowed_tags=None):
    """Boolean array over `chunks`. filters: {column: value or list of values}. allowed_tags: the access tags the
    caller holds. A page is visible if it is tagged 'all' or shares a tag with the caller. Applied BEFORE ranking."""
    m = np.ones(len(chunks), bool)
    for col, val in (filters or {}).items():
        vals = list(val) if isinstance(val, (list, tuple, set)) else [val]
        m &= chunks[col].isin(vals).to_numpy()
    if allowed_tags is not None:
        ok = set(allowed_tags) | {"all"}
        m &= np.array([bool(set(str(t).split(";")) & ok) for t in chunks.access_tags])
    return m

def collapse_to_parents(cand):
    """cand = chunk positions in rank order. Keep the best-ranked child of each page."""
    seen, out = set(), []
    for i in cand:
        pid = chunks.parent_id.iat[i]
        if pid not in seen:
            seen.add(pid); out.append(i)
    return out

# %% [markdown]
# ## 10. Indexes: BM25 + dense HNSW
# BM25 matters most here: users will search voucher numbers, GSTINs, vehicle numbers, amounts. The tokenizer
# keeps IDs whole **and** in a compact form, so `IV/20/2223/0274`, `IV-20-2223-0274` and `iv2022230274` match.

# %%
import bm25s, faiss
def tokenize(t):
    t = t.lower()
    toks = re.findall(r"[a-z0-9ऀ-ॿ]+", t)
    for w in re.findall(r"\S+", t):                       # compact ID / amount forms
        if re.search(r"\d", w) and re.search(r"[/\-.,]", w):
            toks.append(re.sub(r"[^a-z0-9]", "", w))
    return toks

bm25 = bm25s.BM25(k1=1.2, b=0.75)
bm25.index([tokenize(t) for t in chunks.bm25_text])

import gc
for _name in ("embedder", "reranker", "vlm"):      # re-running this cell must not stack a second copy on the GPU
    if _name in globals(): del globals()[_name]
gc.collect()

def gpu_report(tag=""):
    try:
        import torch
        for d in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(d)
            print(f"[gpu{d}{' ' + tag if tag else ''}] used {(total-free)/1e9:.1f} / {total/1e9:.1f} GB "
                  f"(PyTorch allocated {torch.cuda.memory_allocated(d)/1e9:.1f} GB)")
    except Exception as e:
        print("no GPU info:", e)

dense_backend = None
try:
    import torch
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    N_GPU = torch.cuda.device_count()
    EMBED_DEVICE = "cuda:0" if N_GPU else "cpu"
    RERANK_DEVICE = "cuda:1" if N_GPU > 1 else EMBED_DEVICE   # T4 x2: one model per GPU
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer(EMBED_MODEL, device=EMBED_DEVICE)
    embedder.max_seq_length = 1024          # chunks are <1k chars; bge-m3 default 8192 wastes memory
    if N_GPU: embedder.half()
    def embed(texts):
        with torch.inference_mode():
            return embedder.encode(list(texts), batch_size=16, normalize_embeddings=True,
                                   show_progress_bar=False).astype("float32")
    dense_backend = EMBED_MODEL
except Exception as e:
    print("!! Could not load", EMBED_MODEL, "->", repr(e)[:200])
    print("!! FALLBACK: TF-IDF char n-gram + SVD vectors. Plumbing test only, NOT a real dense retriever.")
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.pipeline import make_pipeline
    _tf = make_pipeline(TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True),
                        TruncatedSVD(n_components=min(256, len(chunks) - 1), random_state=0))
    _tf.fit(chunks.embed_text)
    def embed(texts):
        v = _tf.transform(list(texts)).astype("float32")
        return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
    dense_backend = "tfidf-svd-fallback"

# Embedding cache: a chunk whose text (header included) and model are unchanged is not re-embedded.
EMB_F = CACHE / f"emb_{hashlib.md5(str(dense_backend).encode()).hexdigest()[:8]}.npz"
emb_cache = dict(np.load(EMB_F)) if EMB_F.exists() and dense_backend != "tfidf-svd-fallback" else {}
keys = [hashlib.sha1(t.encode()).hexdigest() for t in chunks.embed_text]
miss = [i for i, k in enumerate(keys) if k not in emb_cache]
if miss:
    for i, v in zip(miss, embed(chunks.embed_text.iloc[miss])): emb_cache[keys[i]] = v
vecs = np.stack([emb_cache[k] for k in keys]).astype("float32")
print(f"embedded {len(miss)} new chunks, {len(keys) - len(miss)} from cache")
if dense_backend != "tfidf-svd-fallback":
    np.savez(EMB_F, **{k: emb_cache[k] for k in set(keys)})        # keep only live chunks: deleted files drop out
hnsw = faiss.IndexHNSWFlat(vecs.shape[1], 32, faiss.METRIC_INNER_PRODUCT)   # M=32
hnsw.hnsw.efConstruction = 200
hnsw.add(vecs)
hnsw.hnsw.efSearch = 128
print("dense:", dense_backend, vecs.shape)
gc.collect(); torch.cuda.empty_cache() if "torch" in globals() and torch.cuda.is_available() else None
# At tens of millions of chunks HNSW-flat stops fitting in RAM: move to IVF-PQ / DiskANN / a vector DB.

reranker = None
try:
    from sentence_transformers import CrossEncoder
    reranker = CrossEncoder(RERANK_MODEL, max_length=512, device=RERANK_DEVICE)
    if RERANK_DEVICE.startswith("cuda"):                          # fp16: half the memory, T4-friendly
        (reranker if isinstance(reranker, torch.nn.Module) else reranker.model).half()
except Exception as e:
    print("!! Reranker not loaded ->", repr(e)[:200], "| results will be RRF order")
gpu_report("after loading models")

# %% [markdown]
# ## 10b. File cards (one summary row per voucher)
# A card summarises every page of a voucher: header and Business Central fields, page kinds, whether any page
# contains Devanagari (Hindi), amounts / dates / vehicle numbers found on the supporting pages, and the leading
# text of each page. Search can match the question against cards first and look only inside the top vouchers.
# **Cards only rank and filter.** If no card matches, search falls back to every page, so a bad card never hides
# a page. Whether this loses recall is measured in 12b before anyone relies on it.

# %%
DEVA = re.compile(r"[\u0900-\u097F]")
AMT = re.compile(r"\b\d{1,3}(?:,\d{2,3})*\.\d{2}\b|\b\d{3,7}\b")
VEH = re.compile(r"\b[A-Z]{2}[ -]?\d{2}[ -]?[A-Z]{1,2}[ -]?\d{4}\b")
DATE = re.compile(r"\b\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}\b")
pages_df["has_devanagari"] = pages_df.index_text.map(lambda t: bool(DEVA.search(t or "")))

def build_card(g):
    g = g.sort_values("page")
    r0 = g.iloc[0]
    body = " ".join(re.sub(r"\s+", " ", t or "")[:350] for t in g.index_text)
    sup = " ".join(g[g.page > 1].index_text.fillna(""))
    ents = {"amounts": sorted(set(AMT.findall(sup)))[:40], "dates": sorted(set(DATE.findall(sup)))[:20],
            "vehicles": sorted({re.sub(r"[ -]", "", v) for v in VEH.findall(sup)})[:10]}
    text = (f"Voucher {r0.voucher_no} | {DOC_TYPES.get(r0.voucher_type, r0.voucher_type)} | {len(g)} pages | "
            f"vendor {r0.bc_vendor or ''} | date {r0.bc_posting_date or ''} | amount {r0.bc_amount or ''} | "
            f"kinds {','.join(sorted(set(g.page_kind)))} | {'Hindi present | ' if g.has_devanagari.any() else ''}"
            f"amounts {' '.join(ents['amounts'])} | dates {' '.join(ents['dates'])} | vehicles {' '.join(ents['vehicles'])} | {body}")
    return pd.Series(dict(voucher_no=r0.voucher_no, voucher_type=r0.voucher_type, n_pages=len(g),
                          page_kinds=",".join(sorted(set(g.page_kind))), has_devanagari=bool(g.has_devanagari.any()),
                          min_ocr_conf=float(g.ocr_conf.min()), access_tags=r0.access_tags, card_text=text))

cards = pd.DataFrame([build_card(g) for _, g in pages_df.groupby("voucher_no", sort=False)])
print(len(cards), "file cards;", int(cards.has_devanagari.sum()), "with Devanagari. Pages with Devanagari:",
      [f"{r.file_number} p.{int(r.page)}" for r in pages_df[pages_df.has_devanagari].itertuples()])
card_bm25 = bm25s.BM25(k1=1.2, b=0.75)
card_bm25.index([tokenize(t) for t in cards.card_text])
card_vecs = embed(cards.card_text.str[:3000])

def card_shortlist(query, top=3, rrf_k=60):
    """Voucher numbers of the best-matching cards, or None when nothing matches (caller falls back to all pages)."""
    n = len(cards); pool = min(n, 20)
    ids, sc = card_bm25.retrieve([tokenize(query)], k=pool, show_progress=False, backend_selection="numpy")
    b = {int(i): r for r, (i, s) in enumerate(zip(ids[0], sc[0])) if s > 0}
    d = {int(i): r for r, i in enumerate(np.argsort(-(card_vecs @ embed([query])[0]))[:pool])}
    if not b: return None                       # no keyword evidence: do not trust dense alone, search every page
    fused = {i: 1 / (rrf_k + b.get(i, 99) + 1) + 1 / (rrf_k + d.get(i, 99) + 1) for i in set(b) | set(d)}
    return [cards.voucher_no.iat[i] for i in sorted(fused, key=fused.get, reverse=True)[:top]]

# %% [markdown]
# ## 11. Search: hybrid + RRF + rerank, with citations

# %%
def search(query, k=5, n_candidates=50, rrf_k=60, file_filter=None, mode="hybrid", rerank=True,
           filters=None, allowed_tags=None, collapse=True, use_cards=False, card_top=3):
    """mode: hybrid | bm25 | dense. rerank: apply the cross-encoder to the fused top 30 children.
    filters / allowed_tags: metadata and access restrictions (see visible_mask); pages outside them are never returned.
    collapse: match children, return each page once, with its full text as parent_text.
    With filters, the candidate pool is widened x10 and filtered before fusion is cut. At large scale push the
    filter into the index instead (Qdrant payload filter) so the pool does not need widening."""
    n = len(chunks)
    if use_cards:                       # file cards rank and filter; no card hit = search all pages
        short = card_shortlist(query, top=card_top)
        if short: filters = {**(filters or {}), "voucher_no": short}
    if file_filter: filters = {**(filters or {}), "file_number": norm_id(file_filter)}
    mask = visible_mask(filters, allowed_tags) if (filters or allowed_tags is not None) else None
    pool = min(n, n_candidates * 10) if mask is not None else min(n_candidates, n)
    b_rank, d_rank = {}, {}
    if mode in ("hybrid", "bm25"):
        b_ids, b_sc = bm25.retrieve([tokenize(query)], k=pool, show_progress=False, backend_selection="numpy")
        b_rank = {int(i): r for r, (i, s) in enumerate(zip(b_ids[0], b_sc[0])) if s > 0}
    if mode in ("hybrid", "dense"):
        _, d_ids = hnsw.search(embed([query]), pool)
        d_rank = {int(i): r for r, i in enumerate(d_ids[0]) if i >= 0}
    fused = {}
    for ranks in (b_rank, d_rank):
        for i, r in ranks.items():
            fused[i] = fused.get(i, 0) + 1 / (rrf_k + r + 1)
    cand = sorted(fused, key=fused.get, reverse=True)
    if mask is not None:
        cand = [i for i in cand if mask[i]]
    cand = cand[:30]
    if rerank and reranker is not None and cand:
        pairs = [(query, chunks.embed_text.iat[i][:1500]) for i in cand]   # 1500 chars ~ 512 tokens
        try:
            with torch.inference_mode():
                scores = reranker.predict(pairs, batch_size=8, show_progress_bar=False)
        except torch.cuda.OutOfMemoryError:
            print("!! reranker OOM on GPU; retrying this query on CPU. Run gpu_report() to see what holds memory.")
            torch.cuda.empty_cache()
            (reranker if isinstance(reranker, torch.nn.Module) else reranker.model).float().to("cpu")
            with torch.inference_mode():
                scores = reranker.predict(pairs, batch_size=8, show_progress_bar=False)
        cand = [i for _, i in sorted(zip(scores, cand), reverse=True)]
    if collapse:
        cand = collapse_to_parents(cand)
    res = []
    for i in cand[:k]:
        c = chunks.iloc[i]
        res.append(dict(citation=f"[{c.file_number}, p.{c.page}, {c.file_name}]", file=c.file_name,
                        path=c.rel, page=int(c.page), parent_id=c.parent_id,
                        voucher_type=c.voucher_type, page_kind=c.page_kind, ocr_conf=c.ocr_conf,
                        child_idx=int(c.child_idx), n_children=int(c.n_children),
                        source=c.text_source, flags=c["page_flags"], in_bm25=i in b_rank, in_dense=i in d_rank,
                        snippet=re.sub(r"\s+", " ", c.text)[:160],
                        parent_text=parents.at[c.parent_id, "text"]))
    return res

def show(query, **kw):
    print(f"\nQ: {query}")
    for r in search(query, **kw):
        print(f"  {r['citation']:<28} {r['path']}  (via {'BM25' if r['in_bm25'] else ''}"
              f"{'+' if r['in_bm25'] and r['in_dense'] else ''}{'dense' if r['in_dense'] else ''}, text={r['source']})")
        print(f"      {r['snippet']}")

# %% [markdown]
# ## 12. Sanity-check queries (written knowing the answers; not an evaluation, see 12b)
# Replace these with real questions from the finance team. A RAG that is never measured against the
# questions people actually ask is not done; aim for 100+ question -> (file, page) pairs as an eval set.

# %%
for q in ["IV-20-2223-0274",            # a voucher number: BM25 must find it exactly
          "fastag toll paid wallet vehicle number",
          "hotel stay bill",
          "credit note oil short quantity",
          "freight charges stock transfer",
          "garage repair bill"]:
    show(q, k=3)

# Metadata filter (pre-ranking) and parent text: match a child, read the whole page.
show("receipt for fuel or toll", k=3, filters={"page_kind": ["handwritten_or_faded", "printed"]})
_top = search("garage repair bill", k=1)
if _top:
    print(f"\nParent page for {_top[0]['citation']} ({_top[0]['n_children']} chunk(s), page kind {_top[0]['page_kind']}):")
    print(_top[0]["parent_text"][:400])

# %% [markdown]
# ## 12b. Retrieval evaluation: does the right page come back?
# 22 plain-language questions with the expected file and page(s), spread over printed, handwritten, mixed and
# faded pages. **Caveat:** I wrote them after looking at the pages, so they are kinder than real user questions.
# Replace/extend with questions from the finance team (same columns, upload as `eval_queries.csv`).
# With 22 questions, one question = 4.5 points: read the per-question table, not just the averages.

# %%
EVAL_CSV = next(iter(glob.glob("/kaggle/input/**/eval_queries.csv", recursive=True)), None) if ON_KAGGLE else None
EVAL_CSV = EVAL_CSV or os.environ.get("GP_EVAL_CSV")
if EVAL_CSV and os.path.exists(EVAL_CSV):
    evalq = pd.read_csv(EVAL_CSV)
else:
    # The real question set names employees, vehicles and counterparties, so it is kept out of the repo.
    # Attach eval_queries.csv as a private Kaggle dataset or set GP_EVAL_CSV (format: eval/eval_queries.example.csv).
    print("No eval_queries.csv found: section 12b is skipped.")
    evalq = pd.DataFrame(columns=["question", "expected_file", "expected_pages", "kind"])

def evaluate(mode, rerank, k=5, use_cards=False):
    rows = []
    for _, q in evalq.iterrows():
        exp_pages = {int(x) for x in str(q.expected_pages).split("|")}
        exp_file = norm_id(q.expected_file)
        res = search(q.question, k=k, mode=mode, rerank=rerank, use_cards=use_cards)
        rank = next((i + 1 for i, r in enumerate(res)
                     if r["citation"].startswith(f"[{exp_file},") and r["page"] in exp_pages), None)
        file_rank = next((i + 1 for i, r in enumerate(res) if r["citation"].startswith(f"[{exp_file},")), None)
        rows.append(dict(question=q.question, kind=q.kind, rank=rank, file_rank=file_rank,
                         top1=res[0]["citation"] if res else None))
    df = pd.DataFrame(rows)
    return df, dict(setup=f"{mode}{'+rerank' if rerank else ''}{'+cards' if use_cards else ''}",
                    page_hit1=(df["rank"] == 1).mean(), page_hit5=df["rank"].notna().mean(),
                    MRR=df["rank"].map(lambda r: 1 / r if r == r and r else 0).mean(),
                    file_hit1=(df.file_rank == 1).mean())

setups = [("bm25", False), ("dense", False), ("hybrid", False)] + ([("hybrid", True)] if reranker else [])
summary, per_q = [], {}
for mode, rr in setups:
    df, m = evaluate(mode, rr); summary.append(m); per_q[m["setup"]] = df
print(pd.DataFrame(summary).round(2).to_string(index=False))
# File cards: same questions, search restricted to the top-3 vouchers by card. Recall loss = plain minus card-first.
_m = setups[-1]
_dfc, _mc = evaluate(*_m, use_cards=True); summary.append(_mc); per_q[_mc["setup"]] = _dfc
print(pd.DataFrame(summary).round(2).to_string(index=False))
_plain, _cd = summary[-2], summary[-1]
print(f"\nFile-card recall change on page hit@5: {_cd['page_hit5'] - _plain['page_hit5']:+.2f} "
      f"(adopt cards only if this is ~0 on a real question set). Missed with cards but not without:")
print(per_q[_plain['setup']].assign(card_rank=_dfc['rank'])
      .query("`rank`.notna() and card_rank.isna()")[["question", "rank"]].to_string(index=False))
best = summary[-2]["setup"]
print(f"\nPer question ({best}); rank = position of the right page in the top 5, blank = missed:")
print(per_q[best][["kind", "rank", "top1", "question"]].to_string(index=False))
print("\nBy page kind:")
print(per_q[best].assign(hit5=per_q[best]["rank"].notna()).groupby("kind").hit5.agg(["mean", "count"]).round(2))
pd.DataFrame(summary).to_csv(OUT / "retrieval_eval.csv", index=False)

# %% [markdown]
# ## 13. Save artifacts
# Everything in /kaggle/working is kept when you "Save Version" (Save & Run All). Keep that output private.

# %%
pages_df.drop(columns=["regions"]).to_parquet(OUT / "pages.parquet")
chunks.to_parquet(OUT / "chunks.parquet")
parents.drop(columns=["parent_id"]).to_parquet(OUT / "parents.parquet")
cards.to_parquet(OUT / "cards.parquet")
fileno_df.to_csv(OUT / "file_numbers.csv", index=False)
faiss.write_index(hnsw, str(OUT / "dense_hnsw.faiss"))
bm25.save(str(OUT / "bm25"))
json.dump(dict(dense_backend=dense_backend, reranker=RERANK_MODEL if reranker else None,
               n_files=len(manifest), n_pages=len(pages_df), n_chunks=len(chunks), n_parents=len(parents), n_cards=len(cards), bc_export=BC_CSV,
               built=time.strftime("%Y-%m-%d %H:%M")), open(OUT / "manifest.json", "w"), indent=2)
STATE_F.write_text(json.dumps(now, indent=1))      # file -> content hash, read by section 3 next run
print("saved to", OUT, os.listdir(OUT))

# %% [markdown]
# ## Offline mode (Internet OFF)
# Run this once in a separate notebook **with** Internet ON, then "Save Version" and turn its
# `/kaggle/working/gp-rag-wheels` output into a private dataset named `gp-rag-wheels`. Models: add
# `BAAI/bge-m3` and `BAAI/bge-reranker-v2-m3` via "Add Input > Models" if listed, or `huggingface_hub.snapshot_download`
# them into the same dataset and point `GP_EMBED_MODEL` / `GP_RERANK_MODEL` at the local folders.
# ```
# !pip download --ignore-requires-python -d /kaggle/working/gp-rag-wheels pymupdf rapidocr_onnxruntime==1.4.4 rapid_layout bm25s faiss-cpu wordninja jiwer rapidfuzz sentence-transformers
# from huggingface_hub import snapshot_download
# snapshot_download("BAAI/bge-m3", local_dir="/kaggle/working/gp-rag-wheels/bge-m3")
# snapshot_download("BAAI/bge-reranker-v2-m3", local_dir="/kaggle/working/gp-rag-wheels/bge-reranker-v2-m3")
# ```
