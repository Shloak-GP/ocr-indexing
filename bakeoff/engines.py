"""One OCR engine per process. The orchestrator (bakeoff.py) starts this file once per engine, inside
that engine's own virtualenv, with CUDA_VISIBLE_DEVICES set. A crash or OOM here only loses this engine.

    python engines.py --engine qwen3vl_4b --pages work/pages.csv --out work/runs/qwen3vl_4b.jsonl

Writes one JSON line per page: engine, page_id, text, sec, peak_mb, err. Re-running resumes: pages already
in the output file are skipped. The first line written per process is a {"meta": ...} record with load time
and the library versions actually used, so version drift is visible in the report.
"""
import argparse, csv, json, os, sys, time, traceback

OCR_PROMPT = ("Read all text in this image exactly as written, line by line, top to bottom. Include handwriting, "
              "every number, date, amount and ID, and any Hindi (Devanagari) text in its original script. "
              "Do not translate, summarise, correct or explain. Output plain text only.")

# --------------------------------------------------------------------------------------------------
# Hugging Face vision-language models. All run fp16 + SDPA: T4 (compute 7.5) has no bf16 and no
# FlashAttention-2. 'two_gpu' models are split across both T4s with device_map="balanced".
# --------------------------------------------------------------------------------------------------
HF = {
    "qwen3vl_4b":    dict(repo="Qwen/Qwen3-VL-4B-Instruct", prompt=OCR_PROMPT),
    "qwen3vl_8b":    dict(repo="Qwen/Qwen3-VL-8B-Instruct", prompt=OCR_PROMPT, two_gpu=True),
    "paddleocr_vl":  dict(repo="PaddlePaddle/PaddleOCR-VL-1.6", prompt="OCR:", small=True),
    "glm_ocr":       dict(repo="zai-org/GLM-OCR", prompt="Text Recognition:", small=True),
    "lighton_ocr2":  dict(repo="lightonai/LightOnOCR-2-1B", prompt=None, small=True, max_side=1540,
                          cls="LightOnOcrForConditionalGeneration", proc="LightOnOcrProcessor"),
    "chitrapathak2": dict(repo="krutrim-ai-labs/Chitrapathak-2", system="You are a helpful assistant.",
                          prompt="Perform OCR on this image and transcribe all visible text exactly as it appears."),
}
MAX_NEW_TOKENS = int(os.environ.get("BAKEOFF_MAX_NEW_TOKENS", "2048"))
DEFAULT_MAX_SIDE = int(os.environ.get("BAKEOFF_MAX_SIDE", "1600"))


def _versions(*mods):
    out = {"python": sys.version.split()[0]}
    for m in mods:
        try:
            out[m] = __import__(m).__version__
        except Exception:
            pass
    return out


def _load_image(path, max_side):
    from PIL import Image
    im = Image.open(path).convert("RGB")
    if max(im.size) > max_side:
        s = max_side / max(im.size)
        im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    return im


def _lines_from_boxes(items):
    """items: [(polygon [[x,y]..], text)] -> reading-order text, one line per visual row."""
    rows = []
    for box, txt in items:
        ys = [p[1] for p in box]; xs = [p[0] for p in box]
        rows.append((min(ys), max(ys), min(xs), str(txt)))
    rows.sort(key=lambda t: ((t[0] + t[1]) / 2, t[2]))
    lines, cur = [], []
    for r in rows:
        if cur and abs((r[0] + r[1]) / 2 - (cur[-1][0] + cur[-1][1]) / 2) > 0.5 * max(1, r[1] - r[0]):
            lines.append(cur); cur = []
        cur.append(r)
    if cur:
        lines.append(cur)
    return "\n".join("  ".join(t[3] for t in sorted(ln, key=lambda t: t[2])) for ln in lines)


# ---------------------------------------- CPU engines ---------------------------------------------
def make_pdf_text():
    """The text layer Kodak Capture Pro already embedded in the PDF. Free; the bar every engine must clear."""
    import fitz
    cache = {}
    def run(page):
        d = cache.setdefault(page["pdf"], fitz.open(page["pdf"]))
        return d[int(page["page"]) - 1].get_text()
    return run, _versions("fitz")


def make_tesseract():
    import pytesseract
    from PIL import Image
    langs = os.environ.get("BAKEOFF_TESS_LANGS", "eng+hin")
    def run(page):
        return pytesseract.image_to_string(Image.open(page["image"]), lang=langs, config="--psm 3")
    return run, {**_versions("pytesseract"), "tesseract": str(pytesseract.get_tesseract_version())}


def make_rapidocr():
    import numpy as np
    from PIL import Image
    from rapidocr_onnxruntime import RapidOCR
    eng = RapidOCR()
    def ocr(img):
        res, _ = eng(img)
        return [(r[0], r[1], float(r[2]) if str(r[2]).replace(".", "", 1).isdigit() else 0.0) for r in (res or [])]
    def run(page):
        img = np.array(Image.open(page["image"]).convert("RGB"))
        best = ocr(img)
        score = lambda r: sum(len(t) * c for _, t, c in r)
        if score(best) < 200:                       # sideways scans: same rule as gp_indexing.py
            for k in (1, 3):
                r = ocr(np.ascontiguousarray(np.rot90(img, k)))
                if score(r) > 1.5 * score(best):
                    best = r
        return _lines_from_boxes([(b, t) for b, t, _ in best])
    import rapidocr_onnxruntime
    return run, {"rapidocr_onnxruntime": getattr(rapidocr_onnxruntime, "__version__", "?"), "python": sys.version.split()[0]}


def make_ppocrv5(lang):
    from paddleocr import PaddleOCR
    eng = PaddleOCR(lang=lang, use_doc_orientation_classify=True, use_doc_unwarping=False,
                    use_textline_orientation=True, device="cpu",
                    enable_mkldnn=False)   # paddle 3.3 CPU + oneDNN: "ConvertPirAttribute2RuntimeAttribute not support"
    def run(page):
        out = []
        for r in eng.predict(page["image"]):
            d = r if "rec_texts" in r else r.json.get("res", {})
            polys = d.get("rec_polys", d.get("dt_polys", []))
            out += [([list(map(float, p)) for p in poly], t) for poly, t in zip(polys, d["rec_texts"])]
        return _lines_from_boxes(out)
    return run, _versions("paddleocr", "paddle")


# ---------------------------------------- GPU engines ---------------------------------------------
def make_hf(name):
    import torch, transformers
    cfg = HF[name]
    cls = getattr(transformers, cfg.get("cls", "AutoModelForImageTextToText"))
    proc_cls = getattr(transformers, cfg.get("proc", "AutoProcessor"))
    # Small (~1B) models run fp32: in fp16 on T4 LightOnOCR-2 produced only "!!!!" (overflow). fp32 costs ~4 GB.
    kw = dict(dtype=torch.float32 if cfg.get("small") else torch.float16, attn_implementation="sdpa")
    if cfg.get("two_gpu") and torch.cuda.device_count() >= 2:
        kw.update(device_map="balanced", max_memory={0: "13GiB", 1: "13GiB"})
    else:
        kw.update(device_map={"": 0})
    try:
        model = cls.from_pretrained(cfg["repo"], **kw).eval()
    except ValueError:   # architecture not registered under this Auto class in this transformers version
        alt = getattr(transformers, "AutoModelForMultimodalLM", None) or transformers.AutoModelForCausalLM
        model = alt.from_pretrained(cfg["repo"], **kw).eval()
    proc = proc_cls.from_pretrained(cfg["repo"])
    state = {"dtype": kw["dtype"]}
    max_side = cfg.get("max_side", DEFAULT_MAX_SIDE)

    def generate(img):
        content = [{"type": "image", "image": img}]
        if cfg.get("prompt"):
            content.append({"type": "text", "text": cfg["prompt"]})
        msgs = ([{"role": "system", "content": [{"type": "text", "text": cfg["system"]}]}] if cfg.get("system") else []) \
            + [{"role": "user", "content": content}]
        inputs = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                          return_dict=True, return_tensors="pt")
        dev = model.device
        inputs = {k: (v.to(dev, dtype=state["dtype"]) if v.is_floating_point() else v.to(dev)) for k, v in inputs.items()}
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        return proc.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    def run(page):
        img = _load_image(page["image"], max_side)
        try:
            text = generate(img)
        except RuntimeError as e:
            # fp16 overflow shows up as "probability tensor contains inf/nan". Small models can afford fp32.
            if ("inf" in str(e) or "nan" in str(e)) and cfg.get("small") and state["dtype"] == torch.float16:
                model.float(); state["dtype"] = torch.float32
                text = generate(img)
                page["_note"] = "fp16 overflow; switched to fp32"
            else:
                raise
        if not text.strip() and cfg.get("small") and state["dtype"] == torch.float16:
            model.float(); state["dtype"] = torch.float32
            text = generate(img); page["_note"] = "empty in fp16; retried fp32"
        t = text.strip()
        if len(t) > 50 and max(t.count(c) for c in set(t)) / len(t) > 0.6:
            page["_note"] = f"degenerate output in {state['dtype']} (likely fp16 overflow)"
        return text
    return run, {**_versions("torch", "transformers", "accelerate"), "repo": cfg["repo"], "dtype": str(kw["dtype"])}


def make_chandra2():
    """Reference only: weights are free only below $2M revenue/funding. Uses Datalab's own HF path."""
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from chandra.model.hf import generate_hf
    from chandra.model.schema import BatchInputItem
    from chandra.output import parse_markdown
    model = AutoModelForImageTextToText.from_pretrained("datalab-to/chandra-ocr-2", dtype=torch.float16,
                                                        attn_implementation="sdpa", device_map={"": 0}).eval()
    model.processor = AutoProcessor.from_pretrained("datalab-to/chandra-ocr-2")
    model.processor.tokenizer.padding_side = "left"
    def run(page):
        img = _load_image(page["image"], DEFAULT_MAX_SIDE)
        res = generate_hf([BatchInputItem(image=img, prompt_type="ocr_layout")], model)[0]
        return parse_markdown(res.raw)
    return run, {**_versions("torch", "transformers", "chandra"), "repo": "datalab-to/chandra-ocr-2"}


# ---------------------------------------- Claude API (vision) -------------------------------------
# Sends each page image to the Anthropic API. Needs ANTHROPIC_API_KEY. Company scans leave the building:
# see ocr-bakeoff/OCR_comparison doc, "Data privacy". No server-side fallback on purpose: a fallback would
# silently switch models and the score would no longer belong to the model named in the row.
CLAUDE = {"claude_sonnet55": "claude-sonnet-5-5", "claude_opus55": "claude-opus-5-5"}
PRICE = {"claude-sonnet-5-5": (2.0, 10.0), "claude-opus-5-5": (4.0, 20.0)}   # $ per 1M input/output tokens (Nov 2026)
CLAUDE_MAX_SIDE = int(os.environ.get("BAKEOFF_CLAUDE_MAX_SIDE", "1568"))
CLAUDE_EFFORT = os.environ.get("BAKEOFF_CLAUDE_EFFORT", "medium")


def make_claude(name):
    import base64, io, anthropic
    model = CLAUDE[name]
    client = anthropic.Anthropic(max_retries=6)      # reads ANTHROPIC_API_KEY; retries 429/5xx with backoff
    pin, pout = PRICE[model]

    def run(page):
        im = _load_image(page["image"], CLAUDE_MAX_SIDE)
        b = io.BytesIO(); im.save(b, "PNG")
        msg = client.messages.create(
            model=model, max_tokens=8000,
            thinking={"type": "adaptive"}, output_config={"effort": CLAUDE_EFFORT},
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                             "data": base64.standard_b64encode(b.getvalue()).decode()}},
                {"type": "text", "text": OCR_PROMPT}]}])
        u = msg.usage
        page["_usage"] = {"in": u.input_tokens, "out": u.output_tokens, "stop": msg.stop_reason,
                          "usd": round((u.input_tokens * pin + u.output_tokens * pout) / 1e6, 5)}
        if msg.stop_reason == "refusal":
            page["_note"] = f"refusal: {getattr(msg, 'stop_details', None)}"
        elif msg.stop_reason == "max_tokens":
            page["_note"] = "hit max_tokens"
        return "".join(blk.text for blk in msg.content if blk.type == "text")
    return run, {**_versions("anthropic"), "model": model, "max_side": CLAUDE_MAX_SIDE, "effort": CLAUDE_EFFORT}


FACTORIES = {
    "pdf_text": make_pdf_text,
    "tesseract": make_tesseract,
    "rapidocr": make_rapidocr,
    "ppocrv5_en": lambda: make_ppocrv5("en"),
    "ppocrv5_hi": lambda: make_ppocrv5("hi"),
    "chandra2": make_chandra2,
    **{k: (lambda k=k: make_hf(k)) for k in HF},
    **{k: (lambda k=k: make_claude(k)) for k in CLAUDE},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, choices=sorted(FACTORIES))
    ap.add_argument("--pages", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    pages = list(csv.DictReader(open(a.pages)))
    only = [x.strip() for x in os.environ.get("BAKEOFF_ONLY_PAGES", "").split(",") if x.strip()]
    if only:
        pages = [p for p in pages if p["page_id"] in only]
    if a.limit:
        pages = pages[:a.limit]
    done = set()
    if os.path.exists(a.out):
        for ln in open(a.out):
            try:
                r = json.loads(ln)
                if "page_id" in r and not r.get("err"):
                    done.add(r["page_id"])
            except json.JSONDecodeError:
                pass
    todo = [p for p in pages if p["page_id"] not in done]
    if not todo:
        print(f"[{a.engine}] nothing to do ({len(done)} pages already done)"); return

    try:
        import torch
        has_gpu = torch.cuda.is_available()
        if has_gpu:
            torch.cuda.init()   # torch 2.11: peak-memory calls raise "Invalid device argument" before CUDA init
    except Exception:
        torch, has_gpu = None, False

    def peak_reset():
        if has_gpu:
            for i in range(torch.cuda.device_count()):
                try:
                    torch.cuda.reset_peak_memory_stats(i)
                except RuntimeError:
                    pass   # memory stats are a measurement; never let them stop the run
    def peak_mb():
        if not has_gpu:
            return 0
        try:
            return round(sum(torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())) / 2**20)
        except RuntimeError:
            return -1

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    out = open(a.out, "a", buffering=1)
    peak_reset(); t0 = time.time()
    try:
        run, versions = FACTORIES[a.engine]()
    except Exception as e:
        out.write(json.dumps({"engine": a.engine, "meta": {"load_error": repr(e)[:2000],
                                                           "trace": traceback.format_exc()[-4000:]}}) + "\n")
        print(f"[{a.engine}] LOAD FAILED: {e!r}"); sys.exit(2)
    out.write(json.dumps({"engine": a.engine, "meta": {"load_sec": round(time.time() - t0, 1),
                                                       "load_peak_mb": peak_mb(), "versions": versions,
                                                       "gpus": torch.cuda.device_count() if has_gpu else 0}}) + "\n")
    print(f"[{a.engine}] loaded in {time.time() - t0:.0f}s; {len(todo)} pages to do", flush=True)

    for i, p in enumerate(todo, 1):
        peak_reset(); t = time.time(); err = ""; text = ""
        try:
            text = run(p) or ""
        except Exception as e:
            err = repr(e)[:500]
            if has_gpu and "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
        if has_gpu:
            torch.cuda.synchronize()
        rec = {"engine": a.engine, "page_id": p["page_id"], "text": text, "sec": round(time.time() - t, 2),
               "peak_mb": peak_mb(), "err": err, "note": p.get("_note", "")}
        if "_usage" in p:
            rec["usage"] = p.pop("_usage")
        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[{a.engine}] {i}/{len(todo)} {p['page_id']} {rec['sec']}s {len(text)} chars {err[:80]}", flush=True)


if __name__ == "__main__":
    main()
