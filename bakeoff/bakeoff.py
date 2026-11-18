"""OCR bake-off orchestrator: render pages, build one virtualenv per library stack, run each engine in its
own process (two GPU lanes + one CPU lane), then score everything against typed ground truth.

    python bakeoff.py render --data /kaggle/input --work /kaggle/working/bakeoff
    python bakeoff.py setup  --work ... --engines all
    python bakeoff.py run    --work ... --engines all
    python bakeoff.py score  --work ... --gt gt_fields.csv [--gt-text ocr_ground_truth.csv]

Everything is resumable: re-running 'run' skips pages already written.
"""
import argparse, base64, csv, glob, io, json, math, os, random, re, shutil, subprocess, sys, threading, time
import unicodedata, zlib
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent

# name: (env, gpus, licence, role)   role: "candidate" can be used commercially; "reference" cannot (free).
ENGINES = {
    "pdf_text":      ("classic", 0, "n/a (scanner output)", "baseline"),
    "tesseract":     ("classic", 0, "Apache-2.0", "candidate"),
    "rapidocr":      ("classic", 0, "Apache-2.0", "candidate"),
    "ppocrv5_en":    ("paddle", 0, "Apache-2.0", "candidate"),
    "ppocrv5_hi":    ("paddle", 0, "Apache-2.0", "candidate"),
    "paddleocr_vl":  ("hf5", 1, "Apache-2.0", "candidate"),
    "glm_ocr":       ("hf5", 1, "MIT", "candidate"),
    "lighton_ocr2":  ("hf5", 1, "Apache-2.0", "candidate"),
    "qwen3vl_4b":    ("hf5", 1, "Apache-2.0", "candidate"),
    "qwen3vl_8b":    ("hf5", 2, "Apache-2.0", "candidate"),
    "chitrapathak2": ("hf5", 1, "Krutrim community, research only", "reference"),
    "chandra2":      ("chandra", 1, "OpenRAIL-M, free < $2M revenue", "reference"),
    "claude_sonnet55": ("claude", 0, "paid API; pages sent to Anthropic", "api"),
    "claude_opus55":   ("claude", 0, "paid API; pages sent to Anthropic", "api"),
}
ENVS = {
    # --system-site-packages: reuse Kaggle's torch/CUDA (no 2 GB re-download); the venv's own pins win.
    "classic": [["pymupdf", "pytesseract", "onnxruntime"],
                ["--ignore-requires-python", "rapidocr_onnxruntime==1.4.4"]],   # 1.2.x on Py3.13 = string scores
    "paddle":  [["paddlepaddle>=3.2,<3.4", "paddleocr>=3.3,<4"]],
    "hf5":     [["transformers>=5.0,<6", "accelerate", "pillow"]],
    "chandra": [["chandra-ocr[hf]"]],
    "claude":  [["anthropic>=0.70", "pillow"]],
}
REF_ENGINES = [k for k, v in ENGINES.items() if v[3] == "reference"]


# ------------------------------------------------ render -------------------------------------------
def autocrop(im, margin=24):
    """Crop white margins so a small slip on a big page gets more of each VLM's pixel budget.
    Ignores scanner speckles: a row/column counts only if >0.4% of it is dark."""
    import numpy as np
    g = np.asarray(im.convert("L")) < 170
    rows = np.where(g.sum(1) > 0.004 * g.shape[1])[0]
    cols = np.where(g.sum(0) > 0.004 * g.shape[0])[0]
    if len(rows) < 10 or len(cols) < 10:
        return im
    y0, y1 = max(rows[0] - margin, 0), min(rows[-1] + margin, g.shape[0])
    x0, x1 = max(cols[0] - margin, 0), min(cols[-1] + margin, g.shape[1])
    return im.crop((x0, y0, x1, y1))


def cmd_render(a):
    import fitz
    from PIL import Image
    work = Path(a.work); (work / "pages").mkdir(parents=True, exist_ok=True)
    pdfs = sorted(glob.glob(os.path.join(a.data, "**", "*.pdf"), recursive=True))
    if not pdfs:
        raise SystemExit(f"no PDFs found under {a.data}: attach the dataset first")
    rows = []
    for pdf in pdfs:
        m = re.search(r"(BE\d+/FE\d+/[^/]+\.pdf)$", pdf.replace(os.sep, "/"))
        rel = m.group(1) if m else os.path.basename(pdf)
        stem = Path(pdf).stem
        doc = fitz.open(pdf)
        for i, page in enumerate(doc, 1):
            pix = page.get_pixmap(dpi=a.dpi, colorspace=fitz.csRGB)
            im = Image.frombytes("RGB", (pix.w, pix.h), pix.samples)
            im = autocrop(im)
            fn = work / "pages" / f"{stem}_p{i}.png"
            im.save(fn)
            rows.append(dict(page_id=f"{stem}_p{i}", rel=rel, page=i, image=str(fn), pdf=pdf, w=im.width, h=im.height))
    with open(work / "pages.csv", "w", newline="") as f:
        w = csv.DictWriter(f, rows[0].keys()); w.writeheader(); w.writerows(rows)
    print(f"{len(pdfs)} PDFs, {len(rows)} pages -> {work / 'pages.csv'}")


# ------------------------------------------------ setup --------------------------------------------
def env_python(work, env):
    return str(Path(work) / "envs" / env / "bin" / "python")


def create_env(py):
    """Kaggle's Debian Python has no ensurepip, so 'python -m venv' fails there. Prefer uv (preinstalled on
    Kaggle) with --seed so the env has its own pip. Recreates an env left half-built by an earlier failure."""
    d = Path(py).parent.parent
    if os.path.exists(py) and subprocess.run([py, "-m", "pip", "--version"], capture_output=True).returncode == 0:
        return True
    shutil.rmtree(d, ignore_errors=True)
    tries = []
    if shutil.which("uv"):
        tries.append(["uv", "venv", "-q", "--seed", "--system-site-packages", "--python", sys.executable, str(d)])
    tries.append([sys.executable, "-m", "venv", "--system-site-packages", str(d)])
    for cmd in tries:
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0 and subprocess.run([py, "-m", "pip", "--version"], capture_output=True).returncode == 0:
            return True
        print(f"  {' '.join(cmd[:3])}... failed: {(r.stderr or r.stdout)[-300:]}")
        shutil.rmtree(d, ignore_errors=True)
    return False


def cmd_setup(a):
    engines = pick(a.engines)
    for env in sorted({ENGINES[e][0] for e in engines}):
        py = env_python(a.work, env)
        if not create_env(py):
            print(f"[setup {env}] could not create the virtualenv; engines using it will fail"); continue
        for specs in ENVS[env]:
            t = time.time()
            r = subprocess.run([py, "-m", "pip", "install", "-q", *specs], capture_output=True, text=True)
            print(f"[setup {env}] {' '.join(specs)}: {'ok' if r.returncode == 0 else 'FAILED'} ({time.time() - t:.0f}s)")
            if r.returncode:
                print(r.stderr[-2000:])
        subprocess.run([py, "-c", "import sys; print('  python', sys.version.split()[0])"])


# ------------------------------------------------ run ----------------------------------------------
def pick(spec):
    if spec in ("all", "candidates"):
        # paid API engines only run when named explicitly (they need a key and send pages outside)
        return [e for e in ENGINES if ENGINES[e][3] != "api" and (spec == "all" or ENGINES[e][3] != "reference")]
    names = [s.strip() for s in spec.split(",") if s.strip()]
    bad = [n for n in names if n not in ENGINES]
    if bad:
        raise SystemExit(f"unknown engines {bad}; choose from {list(ENGINES)}")
    return names


def run_one(a, eng, gpus):
    work = Path(a.work); (work / "runs").mkdir(exist_ok=True); (work / "logs").mkdir(exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpus, PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false",
               JAX_PLATFORMS="cpu", HF_HOME=a.hf_home, OMP_NUM_THREADS=str(a.cpu_threads))
    cmd = [env_python(work, ENGINES[eng][0]), str(HERE / "engines.py"), "--engine", eng,
           "--pages", str(work / "pages.csv"), "--out", str(work / "runs" / f"{eng}.jsonl")]
    if a.limit:
        cmd += ["--limit", str(a.limit)]
    t = time.time()
    with open(work / "logs" / f"{eng}.log", "a") as log:
        try:
            rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, timeout=a.timeout_min * 60).returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
    print(f"  {eng:14s} gpus=[{gpus or '-'}] exit={rc} {time.time() - t:6.0f}s  (log: logs/{eng}.log)", flush=True)
    if a.clean_cache and ENGINES[eng][1]:
        for d in glob.glob(os.path.join(a.hf_home, "hub", "models--*")):
            shutil.rmtree(d, ignore_errors=True)   # Kaggle disk is small; each model is downloaded once


def cmd_run(a):
    if not (Path(a.work) / "pages.csv").exists():
        raise SystemExit("pages.csv missing: the render step (cell 4) failed. Fix that first; nothing was run.")
    engines = pick(a.engines)
    cpu = [e for e in engines if ENGINES[e][1] == 0]
    one = [e for e in engines if ENGINES[e][1] == 1]
    two = [e for e in engines if ENGINES[e][1] == 2]
    try:
        import torch
        ngpu = torch.cuda.device_count()
    except ImportError:
        ngpu = 0
    print(f"{ngpu} GPU(s). CPU lane: {cpu}. 1-GPU: {one}. 2-GPU (run last, alone): {two}")
    lanes = [("", cpu)] + ([(str(i), one[i::ngpu]) for i in range(ngpu)] if ngpu else [])
    if a.serial or a.clean_cache:   # clean_cache deletes shared HF cache, so GPU jobs must not overlap
        lanes = [("", cpu)] + ([("0", one)] if ngpu else [])
    threads = [threading.Thread(target=lambda g=g, q=q: [run_one(a, e, g) for e in q]) for g, q in lanes]
    if a.serial:
        for t in threads: t.start(); t.join()
    else:
        for t in threads: t.start()
        for t in threads: t.join()
    for e in two:
        run_one(a, e, ",".join(str(i) for i in range(ngpu)) if ngpu >= 2 else "0")


# ------------------------------------------------ scoring ------------------------------------------
def nfkc(s):
    return unicodedata.normalize("NFKC", s or "")


def number_tokens(text):
    """'Rs.1,250.00' 'Rs 500/-' '₹2,000.00' -> {'1250','500','2000'}. Indian lakh commas handled."""
    out = set()
    for m in re.finditer(r"\d[\d,]*(?:\.\d+)?", nfkc(text)):
        s = m.group().replace(",", "")
        if "." in s:
            whole, frac = s.split(".", 1)
            s = whole if set(frac) <= {"0"} else f"{whole}.{frac.rstrip('0')}"
        s = s.lstrip("0") or "0"
        out.add(s)
    return out


MONTHS = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}


def date_tokens(text):
    """Dates as (day, month[, yy]) keys so 14/10/22, 14-10-2022, 14.10.22 and '14 Oct 2022' all match."""
    t = nfkc(text).lower(); out = set()
    for d, m, y in re.findall(r"\b(\d{1,2})\s*[/.\-]\s*(\d{1,2})(?:\s*[/.\-]\s*(\d{2,4}))?", t):
        if 1 <= int(d) <= 31 and 1 <= int(m) <= 12:
            out.add((int(d), int(m))); y and out.add((int(d), int(m), int(y) % 100))
    for d, mo, y in re.findall(r"\b(\d{1,2})\s*[-/.]?\s*([a-z]{3})[a-z]*[,.]?\s*[-/.]?\s*(\d{2,4})?", t):
        if mo in MONTHS:
            out.add((int(d), MONTHS[mo])); y and out.add((int(d), MONTHS[mo], int(y) % 100))
    for mo, d, y in re.findall(r"\b([a-z]{3})[a-z]*\s+(\d{1,2}),?\s*(\d{4})?", t):   # "October 14, 2022"
        if mo in MONTHS:
            out.add((int(d), MONTHS[mo])); y and out.add((int(d), MONTHS[mo], int(y) % 100))
    return out


def squash(s):
    return re.sub(r"[^0-9A-Zऀ-ॿ]", "", nfkc(s).upper())


def words(s):
    return re.findall(r"[0-9a-zऀ-ॿ]+", nfkc(s).lower())


def split_field(v):
    return [x.strip() for x in (v or "").split(";") if x.strip()]


def field_hits(gt, text):
    """Returns {field: (hits, total)} for one page. Matching is lenient on formatting, strict on content."""
    res = {}
    nums, dates, sq, wd = number_tokens(text), date_tokens(text), squash(text), set(words(text))
    a = split_field(gt.get("amounts"))
    if a: res["amounts"] = (sum(bool(number_tokens(x) & nums) for x in a), len(a))
    d = split_field(gt.get("dates"))
    if d:
        hits = 0
        for x in d:
            want = date_tokens(x)
            full = {k for k in want if len(k) == 3}
            hits += bool((full or want) & dates)
        res["dates"] = (hits, len(d))
    i = split_field(gt.get("ids"))
    if i: res["ids"] = (sum(squash(x) in sq for x in i), len(i))
    n = split_field(gt.get("names"))
    if n:   # a name counts in proportion to its words found (handwritten names are often half-read)
        res["names"] = (sum(sum(w in wd for w in words(x)) / max(1, len(words(x))) for x in n), len(n))
    h = split_field(gt.get("hindi_words"))
    if h: res["hindi_words"] = (sum(nfkc(x) in nfkc(text) for x in h), len(h))
    return res


def cer(ref, hyp):
    r = " ".join(nfkc(ref).split()); h = " ".join(nfkc(hyp).split())
    if not r:
        return None
    prev = list(range(len(h) + 1))
    for i, rc in enumerate(r, 1):
        cur = [i] + [0] * len(h)
        for j, hc in enumerate(h, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rc != hc))
        prev = cur
    return prev[-1] / len(r)


def failure(text, err):
    if err:
        return "error"
    t = (text or "").strip()
    if len(t) < 15:
        return "empty"
    if len(t) > 400 and len(zlib.compress(t.encode())) / len(t.encode()) < 0.12:
        return "loop"    # repeated output compresses far better than real page text
    return ""


def devanagari_share(text):
    t = [c for c in (text or "") if c.isalpha()]
    return sum("ऀ" <= c <= "ॿ" for c in t) / len(t) if t else 0.0


def boot_ci(per_page, n=2000, seed=0):
    """95% CI of hits/total by resampling pages. Wide intervals = not enough ground truth."""
    if not per_page:
        return (None, None)
    rnd = random.Random(seed); vals = []
    for _ in range(n):
        s = [per_page[rnd.randrange(len(per_page))] for _ in per_page]
        tot = sum(t for _, t in s)
        if tot:
            vals.append(sum(h for h, _ in s) / tot)
    vals.sort()
    return (round(vals[int(0.025 * len(vals))], 3), round(vals[int(0.975 * len(vals)) - 1], 3)) if vals else (None, None)


def load_runs(work):
    recs, meta = {}, {}
    for f in glob.glob(str(Path(work) / "runs" / "*.jsonl")):
        for ln in open(f, encoding="utf-8"):
            try:
                r = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if "meta" in r:
                meta[r["engine"]] = r["meta"]
            else:
                recs[(r["engine"], r["page_id"])] = r   # last write wins (a retry overrides an error)
    return recs, meta


def cmd_score(a):
    import pandas as pd
    work = Path(a.work); out = work / "results"; out.mkdir(exist_ok=True)
    pages = pd.read_csv(work / "pages.csv")
    gt = pd.read_csv(a.gt, dtype=str).fillna("") if a.gt and os.path.exists(a.gt) else pd.DataFrame(columns=["rel", "page"])
    if len(gt):
        gt["page"] = gt["page"].astype(int)
        pages = pages.merge(gt, on=["rel", "page"], how="left")
    pages = pages.fillna("")
    if "page_type" not in pages or (pages["page_type"] == "").all():
        pages["page_type"] = "unlabelled"
    gtext = {}
    if a.gt_text and os.path.exists(a.gt_text):
        for _, r in pd.read_csv(a.gt_text, dtype=str).fillna("").iterrows():
            if r.get("ground_truth", "").strip():
                gtext[(r["rel"], int(r["page"]))] = r["ground_truth"]
    recs, meta = load_runs(work)
    engines = sorted({e for e, _ in recs}, key=lambda e: list(ENGINES).index(e) if e in ENGINES else 99)
    if not engines:
        raise SystemExit("no runs found")

    # consensus numbers per page (>=3 digits, produced by at least half of the engines that did not fail)
    consensus = {}
    for pid in pages["page_id"]:
        outs = [number_tokens(recs[(e, pid)]["text"]) for e in engines
                if (e, pid) in recs and not failure(recs[(e, pid)]["text"], recs[(e, pid)]["err"])]
        c = Counter(n for o in outs for n in o if len(n.replace(".", "")) >= 3)
        consensus[pid] = {n for n, k in c.items() if outs and k >= max(2, math.ceil(len(outs) / 2))}

    rows = []
    for _, p in pages.iterrows():
        for e in engines:
            r = recs.get((e, p["page_id"]))
            if r is None:
                continue
            fh = field_hits(p, r["text"])
            cons = consensus[p["page_id"]]
            row = dict(engine=e, page_id=p["page_id"], page_type=p["page_type"], has_hindi=p.get("has_hindi", ""),
                       sec=r["sec"], peak_mb=r["peak_mb"], fail=failure(r["text"], r["err"]), chars=len(r["text"]),
                       dev_share=round(devanagari_share(r["text"]), 3), note=r.get("note", ""),
                       agree=(len(cons & number_tokens(r["text"])) / len(cons)) if cons else None,
                       usd=(r.get("usage") or {}).get("usd"))
            for k, (h, t) in fh.items():
                row[f"{k}_hit"], row[f"{k}_tot"] = h, t
            row["field_hit"] = sum(h for h, _ in fh.values()); row["field_tot"] = sum(t for _, t in fh.values())
            g = gtext.get((p["rel"], int(p["page"])))
            row["cer"] = round(cer(g, r["text"]), 3) if g else None
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out / "per_page.csv", index=False)

    def summarise(g):
        pp = [(h, t) for h, t in zip(g["field_hit"], g["field_tot"]) if t]
        lo, hi = boot_ci(pp)
        tot = sum(t for _, t in pp)
        res = {"pages": len(g), "gt_pages": len(pp), "gt_fields": int(tot),
               "field_recall": round(sum(h for h, _ in pp) / tot, 3) if tot else None, "ci95_low": lo, "ci95_high": hi}
        for k in ["amounts", "dates", "ids", "names", "hindi_words"]:   # same keys every time (pandas stacks otherwise)
            ok = f"{k}_tot" in g and g[f"{k}_tot"].fillna(0).sum()
            res[f"{k}_recall"] = round(g[f"{k}_hit"].fillna(0).sum() / g[f"{k}_tot"].fillna(0).sum(), 3) if ok else None
        res["cer_mean"] = round(g["cer"].dropna().mean(), 3) if g["cer"].notna().any() else None
        res["cer_pages"] = int(g["cer"].notna().sum())
        res["fail_rate"] = round((g["fail"] != "").mean(), 3)
        res["agree_numbers"] = round(g["agree"].dropna().mean(), 3) if g["agree"].notna().any() else None
        res["sec_median"] = round(g["sec"].median(), 2); res["sec_p90"] = round(g["sec"].quantile(0.9), 2)
        res["peak_gpu_mb"] = int(g["peak_mb"].max())
        has_usd = "usd" in g and g["usd"].notna().any()
        res["usd_per_page"] = round(g["usd"].mean(), 4) if has_usd else None
        res["usd_total"] = round(g["usd"].sum(), 3) if has_usd else None
        return pd.Series(res)

    summ = df.groupby("engine").apply(summarise).reset_index()
    summ["licence"] = summ["engine"].map(lambda e: ENGINES.get(e, ("", 0, "?", ""))[2])
    summ["role"] = summ["engine"].map(lambda e: ENGINES.get(e, ("", 0, "", "?"))[3])
    summ["load_sec"] = summ["engine"].map(lambda e: meta.get(e, {}).get("load_sec"))
    summ = summ.sort_values(["field_recall", "agree_numbers", "sec_median"], ascending=[False, False, True], na_position="last")
    summ.to_csv(out / "summary.csv", index=False)
    by_type = df.groupby(["page_type", "engine"]).apply(summarise).reset_index()
    by_type.to_csv(out / "by_page_type.csv", index=False)
    hindi = df[(df["has_hindi"] == "yes") | (df["page_type"] == "hindi")]
    hindi_tbl = hindi.groupby("engine").agg(pages=("page_id", "nunique"), devanagari_share=("dev_share", "mean"),
                                            hindi_hit=("hindi_words_hit", "sum") if "hindi_words_hit" in hindi else ("dev_share", "size"),
                                            fail_rate=("fail", lambda s: (s != "").mean())).reset_index() if len(hindi) else pd.DataFrame()
    if len(hindi_tbl):
        hindi_tbl.to_csv(out / "hindi.csv", index=False)

    # Routing: best candidate engine per page type (accuracy first, then speed), with the cost of that mix.
    route = []
    cand = by_type[by_type["engine"].map(lambda e: ENGINES.get(e, ("", 0, "", ""))[3] == "candidate")]
    for t, g in cand.groupby("page_type"):
        key = "field_recall" if g["field_recall"].notna().any() else "agree_numbers"
        g = g.sort_values([key, "sec_median"], ascending=[False, True])
        b = g.iloc[0]
        route.append(dict(page_type=t, best=b["engine"], ranked_by=key, score=b[key], sec_median=b["sec_median"],
                          pages=int(b["pages"]), runner_up=g.iloc[1]["engine"] if len(g) > 1 else ""))
    route = pd.DataFrame(route)
    if len(route):
        sec = (route["sec_median"] * route["pages"]).sum() / route["pages"].sum()
        route.to_csv(out / "routing.csv", index=False)

    with open(out / "report.md", "w") as f:
        f.write("# OCR bake-off results\n\n")
        f.write(f"{pages['page_id'].nunique()} pages, {len(engines)} engines. Pages with typed field ground truth: "
                f"{int((summ['gt_pages'].max() if len(summ) else 0))}. Pages with full-text ground truth: {len(gtext)}.\n\n")
        f.write("**Read this first.** field_recall is the share of typed amounts/dates/IDs/names found in the output. "
                "ci95 is its 95% range from resampling pages: if two engines' ranges overlap, the sample can't separate "
                "them. agree_numbers is agreement with the other engines' numbers, NOT accuracy (engines can agree on a "
                "wrong number). T4 seconds are for ranking only; production GPUs are several times faster.\n\n")
        f.write("## Overall\n\n" + md(summ) + "\n\n")
        f.write("## By page type\n\n" + md(by_type) + "\n\n")
        if len(hindi_tbl):
            f.write("## Hindi pages\n\n" + md(hindi_tbl) + "\n\n")
        if len(route):
            f.write(f"## Routing (best licence-clean engine per page type)\n\nWeighted median time with this mix: {sec:.1f} s/page "
                    f"= {sec * 100000 / 3600 / 24:.1f} T4-days per 100k pages on one GPU (mix taken from this sample, "
                    f"which is too small to trust).\n\n" + md(route) + "\n\n")
        f.write("## Library versions actually used\n\n")
        for e, m in meta.items():
            f.write(f"- {e}: {json.dumps(m.get('versions', m))[:400]}\n")
    write_compare_html(work, pages, engines, recs, out / "compare.html")
    print((out / "report.md").read_text()[:6000])
    print(f"\nWritten: {out}/report.md, summary.csv, by_page_type.csv, per_page.csv, routing.csv, compare.html")


def md(df):
    try:
        return df.to_markdown(index=False)
    except ImportError:          # tabulate missing
        return "```\n" + df.to_string(index=False) + "\n```"


def write_compare_html(work, pages, engines, recs, path):
    """One section per page: the image next to every engine's text. Use it to type ground truth and eyeball errors."""
    from PIL import Image
    import html
    parts = ["<html><meta charset='utf-8'><style>body{font-family:sans-serif} .p{display:flex;gap:12px;border-top:2px solid #333;"
             "padding:8px} img{max-width:620px} .o{flex:1;overflow:auto;max-height:900px} pre{white-space:pre-wrap;"
             "font-size:12px;background:#f6f6f6;padding:6px} h4{margin:6px 0}</style><body><h2>OCR bake-off: side by side</h2>"]
    for _, p in pages.iterrows():
        im = Image.open(p["image"]).convert("L"); im.thumbnail((1100, 1500))
        b = io.BytesIO(); im.save(b, "JPEG", quality=70)
        gt = "; ".join(f"{k}: {p.get(k, '')}" for k in ["amounts", "dates", "ids", "names"] if p.get(k, ""))
        parts.append(f"<div class='p'><div><h3>{p['page_id']} ({p.get('page_type', '')})</h3>"
                     f"<img src='data:image/jpeg;base64,{base64.b64encode(b.getvalue()).decode()}'>"
                     f"<p><b>Ground truth:</b> {html.escape(gt) or '<i>not typed yet</i>'}</p></div><div class='o'>")
        for e in engines:
            r = recs.get((e, p["page_id"]))
            if r:
                parts.append(f"<h4>{e} ({r['sec']}s){' ERROR ' + html.escape(r['err']) if r['err'] else ''}</h4>"
                             f"<pre>{html.escape(r['text'][:6000])}</pre>")
        parts.append("</div></div>")
    Path(path).write_text("".join(parts) + "</body></html>", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for n in ["render", "setup", "run", "score"]:
        s = sub.add_parser(n)
        s.add_argument("--work", default="/kaggle/working/bakeoff")
        s.add_argument("--engines", default="all")
    sub.choices["render"].add_argument("--data", default="/kaggle/input")
    sub.choices["render"].add_argument("--dpi", type=int, default=200)
    r = sub.choices["run"]
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--timeout-min", type=int, default=120)
    r.add_argument("--hf-home", default="/tmp/hf")
    r.add_argument("--cpu-threads", type=int, default=2)
    r.add_argument("--serial", action="store_true", help="one engine at a time: cleaner timings, slower")
    r.add_argument("--clean-cache", action="store_true", help="delete each model after use (disk-tight)")
    s = sub.choices["score"]
    s.add_argument("--gt", default=str(HERE / "gt_fields.csv"))
    s.add_argument("--gt-text", default="")
    a = ap.parse_args()
    {"render": cmd_render, "setup": cmd_setup, "run": cmd_run, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    main()
