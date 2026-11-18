"""Hand-check list for the OCR answer key, and rescoring after the check.

  python disagreements.py build  --runs DIR [DIR ...] --gt gt_fields.csv --out check.xlsx
  python disagreements.py rescore --runs DIR [DIR ...] --gt gt_fields.csv --check check.xlsx --out rescore.xlsx

build:   one row per answer-key value that at least one of the 4 engines does NOT find, plus one row per
         amount-like number that some engines read but others don't and that isn't in the key
         ("precision" rows: either the key missed a value or an engine invented / misread one).
         Values that all 4 engines find are left out (counted on the Summary tab).
rescore: reads your Verdict / Scan says columns, builds a corrected key (gt_fields_corrected.csv next to --out)
         and recomputes field recall per engine with bakeoff.py's own matching.

Verdict values (column H):
  key ok        the key value is what the scan says
  key wrong     the key is wrong; put the real value in "Scan says"
  not on page   remove the value from the key
  unreadable    remove it from the key (nobody can be scored on it)
  add to key    (precision rows only) the number is really on the page; it is added to the key
  not real      (precision rows only) the engine(s) that wrote it invented / misread it
"""
import argparse, csv, json, re, sys
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from bakeoff import (date_tokens, failure, field_hits, nfkc, number_tokens, split_field,  # noqa: E402
                     squash, words)

ENGINES = ["claude_sonnet55", "lighton_ocr2", "qwen3vl_8b", "qwen3vl_4b"]
LABEL = {"claude_sonnet55": "Claude Sonnet 5.5", "lighton_ocr2": "LightOnOCR-2",
         "qwen3vl_8b": "Qwen3-VL-8B", "qwen3vl_4b": "Qwen3-VL-4B"}
FIELDS = ["amounts", "dates", "ids", "names"]


def load_runs(dirs):
    recs = {}
    for d in dirs:
        for f in sorted(Path(d).rglob("*.jsonl")):
            for ln in open(f, encoding="utf-8"):
                try:
                    r = json.loads(ln)
                except json.JSONDecodeError:
                    continue
                if "meta" not in r and r.get("engine") in ENGINES:
                    recs[(r["engine"], r["page_id"])] = r
    return recs


def page_id(rel, page):
    return f"{Path(rel).stem}_p{int(page)}"


# ---------- what did the engine write instead? (closest candidate on the page) ----------
def num_spans(text):
    return [m.group() for m in re.finditer(r"\d[\d,]*(?:\.\d+)?", nfkc(text))]


def date_spans(text):
    t = nfkc(text)
    pats = [r"\b\d{1,2}\s*[/.\-]\s*\d{1,2}(?:\s*[/.\-]\s*\d{2,4})?",
            r"\b\d{1,2}\s*[-/.]?\s*[A-Za-z]{3}[a-z]*[,.]?\s*[-/.]?\s*(?:\d{2,4})?",
            r"\b[A-Za-z]{3}[a-z]*\s+\d{1,2},?\s*(?:\d{4})?"]
    out = []
    for p in pats:
        out += [m.group().strip() for m in re.finditer(p, t) if date_tokens(m.group())]
    return out


def best(target, cands, key=lambda s: s):
    tk = key(target)
    scored = [(SequenceMatcher(None, tk, key(c)).ratio(), c) for c in cands if key(c)]
    if not scored:
        return "", 0.0
    r, c = max(scored)
    return c, r


def closest(field, value, text):
    """Best guess at the engine's reading of `value`; '' if nothing similar."""
    if field == "amounts":
        c, r = best(value, num_spans(text), key=lambda s: s.replace(",", "").split(".")[0])
    elif field == "dates":
        c, r = best(value, date_spans(text), key=lambda s: re.sub(r"\W", "", s.lower()))
    elif field == "ids":
        toks = re.findall(r"[A-Za-z0-9][A-Za-z0-9/\-. ]{2,40}", nfkc(text))
        n = len(squash(value))
        cands = set()
        for t in toks:
            s = t.strip()
            for i in range(len(s)):
                for L in (n - 2, n, n + 2):
                    if L > 2 and i + L <= len(s):
                        cands.add(s[i:i + L])
        c, r = best(value, cands, key=squash)
    else:  # names: window of the same number of words
        ws = re.findall(r"\S+", nfkc(text)); k = max(1, len(words(value)))
        c, r = best(value, [" ".join(ws[i:i + k]) for i in range(len(ws))], key=lambda s: " ".join(words(s)))
    return (c, r) if r >= 0.5 else ("", r)


def engine_cell(field, value, rec):
    """('found' | 'partial' | 'missing' | 'page failed', text shown in the sheet)"""
    if rec is None:
        return "no output", "NO OUTPUT"
    text, fail = rec.get("text") or "", failure(rec.get("text"), rec.get("err"))
    tag = f"[page {fail}] " if fail else ""   # scored as-is, same as bakeoff.py (a looped page can still hold values)
    h, t = field_hits({field: value}, text)[field]
    if h >= t:
        return "found", tag + "✓"
    c, _ = closest(field, value, text)
    if field == "names" and h > 0:
        return "partial", tag + (f"partial: {c}" if c else "partial")
    return "missing", tag + (c if c else "(not found)")


def amount_like(text):
    """Number tokens with at least 3 digits before any decimal point."""
    out = {}
    for s in num_spans(text):
        tok = next(iter(number_tokens(s)), "")
        if not tok or len(tok.split(".")[0]) < 3:   # skip 0, 1.0, "04," etc.: too short to check usefully
            continue
        out.setdefault(tok, s)
    return out


# ---------------------------------------------- build ----------------------------------------------
def cmd_build(a):
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.worksheet.datavalidation import DataValidation

    gt = list(csv.DictReader(open(a.gt, encoding="utf-8")))
    recs = load_runs(a.runs)
    have = [e for e in ENGINES if any(k[0] == e for k in recs)]
    missing_eng = [e for e in ENGINES if e not in have]
    rows, agree, per_field = [], 0, Counter()
    for g in gt:
        pid = page_id(g["rel"], g["page"])
        pr = {e: recs.get((e, pid)) for e in have}
        key_digits = set()
        for f in FIELDS:
            for v in split_field(g.get(f)):
                key_digits |= number_tokens(v)
                cells = {e: engine_cell(f, v, pr[e]) for e in have}
                if all(s == "found" for s, _ in cells.values()) and not missing_eng:
                    agree += 1
                    continue
                n_found = sum(s == "found" for s, _ in cells.values())
                rows.append({"kind": "key value", "file": g["rel"], "page": int(g["page"]), "page_type": g["page_type"],
                             "field": f, "key": v, "cells": {e: c[1] for e, c in cells.items()},
                             "why": f"{n_found}/{len(have)} engines find the key value"})
                per_field[f] += 1
        # precision rows: amount-like numbers read by some engines, not all, and not in the key
        ok = {e: pr[e] for e in have if pr[e] and failure(pr[e].get("text"), pr[e].get("err")) in ("", "loop")}
        seen = {e: amount_like(r["text"]) for e, r in ok.items()}
        allnums = set().union(*seen.values()) if seen else set()
        for tok in sorted(allnums, key=lambda s: float(s) if s.replace(".", "").isdigit() else 0):
            who = [e for e in seen if tok in seen[e]]
            if tok in key_digits or len(who) == len(seen):
                continue
            if any(tok in number_tokens(x) for x in [g.get("ids", ""), g.get("dates", "")]):
                continue
            cells = {}
            for e in have:
                if e not in ok:
                    cells[e] = "PAGE FAILED / no output"
                elif e in who:
                    cells[e] = f"wrote {seen[e][tok]}"
                else:
                    c, _ = closest("amounts", tok, ok[e]["text"])
                    cells[e] = f"– (nearest: {c})" if c else "–"
            rows.append({"kind": "not in key", "file": g["rel"], "page": int(g["page"]), "page_type": g["page_type"],
                         "field": "number", "key": f"(not in key) {tok}", "n_who": len(who), "cells": cells,
                         "why": f"written by {len(who)}/{len(seen)} engines: " + ", ".join(LABEL[e] for e in who)})
    rows.sort(key=lambda r: (r["file"], r["page"], r["kind"] != "key value", FIELDS.index(r["field"]) if r["field"] in FIELDS else 9))

    rows1 = [r for r in rows if r.get("n_who") == 1]
    rows = [r for r in rows if r.get("n_who") != 1]
    wb = Workbook(); ws = wb.active; ws.title = "Check list"
    head = ["#", "File", "Page", "Page type", "Row type", "Field", "Answer key value"] + [LABEL[e] for e in ENGINES] + \
           ["Verdict (from scan)", "Scan says (if key wrong / add)", "Why listed", "Your notes"]
    ws.append(head)
    for i, r in enumerate(rows, 1):
        ws.append([i, r["file"], r["page"], r["page_type"], r["kind"], r["field"], r["key"]] +
                  [r["cells"].get(e, "ENGINE OUTPUT NOT AVAILABLE") for e in ENGINES] + ["", "", r["why"], ""])
    bold = Font(bold=True); fill = PatternFill("solid", fgColor="FFF2CC"); grey = PatternFill("solid", fgColor="EEEEEE")
    for c in ws[1]:
        c.font = bold; c.alignment = Alignment(wrap_text=True, vertical="top")
    vcol, scol = 12, 13
    for row in ws.iter_rows(min_row=2):
        for c in row:
            c.alignment = Alignment(wrap_text=True, vertical="top")
        row[vcol - 1].fill = fill; row[scol - 1].fill = fill
        if row[4].value == "not in key":
            for c in row[:7]:
                c.fill = grey
    widths = [5, 44, 6, 12, 11, 12, 22, 22, 22, 22, 22, 16, 22, 34, 24]
    for i, w in enumerate(widths):
        ws.column_dimensions[ws.cell(1, i + 1).column_letter].width = w
    ws.freeze_panes = "A2"; ws.auto_filter.ref = ws.dimensions
    dv = DataValidation(type="list", formula1='"key ok,key wrong,not on page,unreadable,add to key,not real"', allow_blank=True)
    ws.add_data_validation(dv); dv.add(f"L2:L{len(rows) + 1}")
    # numbers only one engine wrote: almost always that engine's misread. Optional, on their own tab.
    ws2 = wb.create_sheet("Optional - 1 engine numbers"); ws2.append(head)
    for i, r in enumerate(rows1, 1):
        ws2.append([i, r["file"], r["page"], r["page_type"], r["kind"], r["field"], r["key"]] +
                   [r["cells"].get(e, "ENGINE OUTPUT NOT AVAILABLE") for e in ENGINES] + ["", "", r["why"], ""])
    for c in ws2[1]:
        c.font = bold; c.alignment = Alignment(wrap_text=True, vertical="top")
    for row in ws2.iter_rows(min_row=2):
        for c in row:
            c.alignment = Alignment(wrap_text=True, vertical="top")
        row[vcol - 1].fill = fill; row[scol - 1].fill = fill
    for i, w in enumerate(widths):
        ws2.column_dimensions[ws2.cell(1, i + 1).column_letter].width = w
    ws2.freeze_panes = "A2"; ws2.auto_filter.ref = ws2.dimensions
    dv2 = DataValidation(type="list", formula1='"add to key,not real"', allow_blank=True)
    ws2.add_data_validation(dv2); dv2.add(f"L2:L{len(rows1) + 1}")

    s = wb.create_sheet("How to use")
    help_lines = [
        "One row per value to check. Open the scan at File + Page and pick a Verdict (dropdown in column L).",
        "✓ = engine found the key value. Text = the closest thing the engine wrote instead. PAGE FAILED = engine looped / returned nothing.",
        "Grey rows = numbers some engines wrote that are NOT in the key. Either the key missed it, or an engine misread / invented it.",
        "",
        "Verdicts:",
        "  key ok       the key value is right",
        "  key wrong    the key is wrong; type the real value in 'Scan says'",
        "  not on page  remove the value from the key",
        "  unreadable   remove it from the key (nobody can be scored on it)",
        "  add to key   (grey rows) the number is really on the page; it gets added to the key",
        "  not real     (grey rows) the engines that wrote it misread / invented it",
        "",
        "Blank verdict on a white row = treated as 'key ok'. Blank on a grey row = ignored.",
        "Send the filled file back and recall is recomputed for all 4 engines against the corrected key.",
    ]
    for ln in help_lines:
        s.append([ln])
    s.column_dimensions["A"].width = 130

    sm = wb.create_sheet("Summary")
    sm.append(["Key values where all 4 engines agree with the key (not listed)", agree])
    sm.append(["Key values listed (at least one engine misses)", sum(r["kind"] == "key value" for r in rows)])
    for f in FIELDS:
        sm.append([f"  of which {f}", per_field[f]])
    sm.append(["Numbers not in key, written by 2+ engines but not all (grey rows, Check list tab)", sum(r["kind"] == "not in key" for r in rows)])
    sm.append(["Numbers only one engine wrote (Optional tab; usually that engine's misread)", len(rows1)])
    sm.append(["Engines with outputs", ", ".join(LABEL[e] for e in have)])
    sm.append(["Engines MISSING outputs", ", ".join(LABEL[e] for e in missing_eng) or "none"])
    sm.column_dimensions["A"].width = 70
    wb.save(a.out)
    print(f"{len(rows1)} one-engine numbers on optional tab; {len(rows)} rows ({sum(r['kind'] == 'key value' for r in rows)} key values, "
          f"{sum(r['kind'] != 'key value' for r in rows)} not-in-key numbers); {agree} key values all engines agree on.")
    print("engines with outputs:", have, "| missing:", missing_eng)


# ---------------------------------------------- rescore --------------------------------------------
def cmd_rescore(a):
    from openpyxl import Workbook, load_workbook
    gt = list(csv.DictReader(open(a.gt, encoding="utf-8")))
    wbc = load_workbook(a.check)
    sheets = [s for s in wbc.worksheets if s.title in ("Check list", "Optional - 1 engine numbers")]
    hdr = [c.value for c in sheets[0][1]]
    ix = {h: i for i, h in enumerate(hdr)}
    edits = Counter()
    byp = {(g["rel"], int(g["page"])): g for g in gt}
    allrows = [row for s in sheets for row in s.iter_rows(min_row=2, values_only=True)]
    for row in allrows:
        verdict = (row[ix["Verdict (from scan)"]] or "").strip().lower()
        said = str(row[ix["Scan says (if key wrong / add)"]] or "").strip()
        g = byp.get((row[ix["File"]], int(row[ix["Page"]])))
        if g is None or not verdict or verdict == "key ok" or verdict == "not real":
            continue
        field, key = row[ix["Field"]], str(row[ix["Answer key value"]])
        if row[ix["Row type"]] == "not in key":
            if verdict == "add to key":
                num = said or str(row[ix["Answer key value"]]).split()[-1]
                g["amounts"] = ";".join(split_field(g["amounts"]) + [num]); edits["added"] += 1
            continue
        vals = split_field(g[field])
        if key not in vals:
            print("warning: key value not found in gt, skipped:", row[ix["File"]], row[ix["Page"]], field, key); continue
        if verdict == "key wrong" and said:
            vals[vals.index(key)] = said; edits["corrected"] += 1
        elif verdict in ("not on page", "unreadable"):
            vals.remove(key); edits["removed"] += 1
        g[field] = ";".join(vals)
    outp = Path(a.out); corr = outp.with_name("gt_fields_corrected.csv")
    with open(corr, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(gt[0].keys())); w.writeheader(); w.writerows(gt)

    recs = load_runs(a.runs)
    orig = list(csv.DictReader(open(a.gt, encoding="utf-8")))
    wb = Workbook(); s = wb.active; s.title = "Recall"
    s.append(["Engine", "Recall (original key)", "Recall (corrected key)", "Change", "Pages", "Failed pages"])
    for e in ENGINES:
        res = []
        for key_rows in (orig, gt):
            hit = tot = 0
            for g in key_rows:
                r = recs.get((e, page_id(g["rel"], g["page"])))
                if r is None:
                    continue
                txt = r.get("text") or ""   # same as bakeoff.py: failed pages scored on whatever they output
                for f, (h, t) in field_hits({k: g.get(k) for k in FIELDS + ["hindi_words"]}, txt).items():
                    hit += h; tot += t
            res.append(hit / tot if tot else None)
        pages = sum(1 for g in gt if (e, page_id(g["rel"], g["page"])) in recs)
        fails = sum(1 for g in gt if (r := recs.get((e, page_id(g["rel"], g["page"])))) and failure(r.get("text"), r.get("err")))
        s.append([LABEL[e]] + [round(x, 3) if x is not None else "no output" for x in res] +
                 [round(res[1] - res[0], 3) if None not in res else "", pages, fails])
    s.append([]); s.append([f"Key edits: {dict(edits)}"]); s.append([f"Corrected key: {corr.name}"])
    wb.save(outp); print(f"edits {dict(edits)} -> {corr}, {outp}")


def main():
    p = argparse.ArgumentParser(); sub = p.add_subparsers(dest="cmd", required=True)
    for c in ("build", "rescore"):
        q = sub.add_parser(c); q.add_argument("--runs", nargs="+", required=True); q.add_argument("--gt", required=True)
        q.add_argument("--out", required=True)
        if c == "rescore":
            q.add_argument("--check", required=True)
    a = p.parse_args(); {"build": cmd_build, "rescore": cmd_rescore}[a.cmd](a)


if __name__ == "__main__":
    main()
