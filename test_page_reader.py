"""Unit test of the reader routing rules in gp_indexing.py section 6 (no GPU, no models)."""
import os, re, zlib
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "gp_indexing.py")).read()
a = src.index("# --- reader core"); b = src.index("# --- end reader core ---")
ns = dict(zlib=zlib, re=re); exec(src[a:b], ns)
rf, nf, pt, dl, hh = ns["reader_failure"], ns["needs_fallback"], ns["pick_text"], ns["deloop"], ns["has_hindi"]

good = "Voucher IV/20/2223/0274 Date 22-Feb-22 Amount Rs 4,500.00 Paid to Ramesh Kumar for diesel"
hindi = good + "\nरवि ऑटोमोबाइल सर्विसेज कैश मेमो"
assert rf(good) == "" and rf("") == "empty" and rf(None) == "empty" and rf("ab") == "empty"
assert rf("!" * 300) == "degenerate"
assert rf("total 100 " * 200) == "loop"                       # repetition
assert rf("\n".join(f"line {i} item {i*37} amt {i*91}" for i in range(60))) == ""   # long real-looking text passes
assert nf(good, []) == "" and nf(good, ["check_script"]) == "hindi page" and nf(good, ["check_script"], False) == ""
assert hh(hindi) and not hh(good) and nf(hindi, []) == "hindi page"   # Devanagari in the primary read routes it
assert nf("", []) == "primary empty"
assert pt(good, None, "") == (good, "primary", "")
fb = "Fallback read: " + good
assert pt("", fb, "primary empty")[:2] == (fb, "fallback")
assert pt("", "", "primary empty")[0] is None                 # nothing to salvage -> RapidOCR in section 7
assert pt(good, fb, "hindi page")[:2] == (fb, "fallback")
assert pt(good, "", "hindi page")[:2] == (good, "primary")    # hindi fallback failed: keep primary
looped = "GP Petroleums Ltd.\nOutward No. 1818\nDate: 15-09-22\nNotes:\n" + "- 2008\n" * 300 + "- 20"
assert rf(looped) == "loop"
d = dl(looped)
assert rf(d) == "" and "Outward No. 1818" in d and d.count("2008") == 1, d
assert dl("a\nb\nc\nd") == "a\nb\nc\nd"                       # nothing repeated: unchanged
blk = "my good hand\nRs 20\n"
assert dl("Head 1\n" + blk * 50).count("my good hand") == 1    # 2-line period
t, r, n = pt(looped, "x" * 3 + "\n" + "y" * 900, "primary loop")
assert r == "primary_deloop" and "1818" in t, (r, n)         # both failed: de-looped primary kept
print("test_page_reader: all assertions passed")

# voucher splitting (section 5)
s2 = src[src.index("ID_RE = re.compile"):src.index("def find_file_number")] + src[src.index("def voucher_start"):src.index("def looks_like_id")]
ns2 = dict(re=re); exec(s2, ns2); vs = ns2["voucher_start"]
assert vs("GP Petroleums Limited\nEMPLOYEEVoucher\nVoucherNo.:STF20/2627/0733  Date:16-09-26", "", 4) == "STF/20/2627/0733"
assert vs("Document No: NR/RPO/2223/1013", "", 1) == "NR/RPO/2223/1013"
assert vs("e-Way Bill DocNo.:TaxInvoice-AP/20/2627/0548 Date", "", 32) is None   # supplier doc inside a batch
assert vs("Tax Invoice Doc No.: GPPL/PO/2526/1622", "", 9) is None
print("voucher_start: all assertions passed")

# ---- show-through stamp pages (Nov 17) ----
import numpy as np
a2 = src.index("# --- show-through core"); b2 = src.index("# --- end show-through core ---")
ns2 = dict(np=np); exec(src[a2:b2], ns2)
is_st, supp = ns2["is_show_through_stamp_page"], ns2["suppress_show_through"]
rng = np.random.default_rng(0)
blank = np.full((200, 150, 3), 245, np.uint8)
page = blank.copy(); page[:, :, :] = 245
back = blank.copy(); back[20:60, 20:100] = (205, 205, 205)                 # faint grey mirrored text
back[140:170, 30:110] = (140, 90, 190)                                      # purple stamp (coloured, mid-dark)
assert is_st(back)
assert not is_st(blank)                                                     # blank: no stamp
text = blank.copy(); text[10:190:6, 10:140] = (20, 20, 20)                  # real printed page: lots of dark ink
assert not is_st(text)
grey = blank.copy(); grey[20:60, 20:100] = (205, 205, 205)
assert not is_st(grey)                                                      # faint text but no stamp: leave alone
c = supp(back)
assert (c[20:60, 20:100] == 255).all()                                      # show-through removed
assert (c[140:170, 30:110] == (140, 90, 190)).all()                         # stamp kept
assert (supp(text)[10:190:6, 10:140] == (20, 20, 20)).all()                 # dark ink kept
print("show-through tests OK")

# ---- stamp boxes (Nov 18) ----
sb = ns2["stamp_boxes"]
two = np.full((600, 400, 3), 245, np.uint8)
two[50:150, 40:200] = (140, 90, 190); two[400:480, 100:330] = (150, 80, 180)   # two separate purple stamps
two[250:260, 10:390] = (205, 205, 205)                                          # grey show-through must not become a box
bx = sb(two, pad=5)
assert len(bx) == 2, bx
assert bx[0][1] <= 50 and bx[0][3] >= 150 and bx[1][1] <= 400 and bx[1][3] >= 480
assert sb(blank) == []
print("stamp_boxes tests OK")
