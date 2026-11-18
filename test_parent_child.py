import re, os, sys, types
import numpy as np, pandas as pd
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "gp_indexing.py")).read()
def cell(start_marker, end_marker):
    a = src.index(start_marker); b = src.index(end_marker, a)
    return src[a:b]
sec9 = cell("# %%\ndef split_page", "# %% [markdown]")
srch = cell("def search(query", "def show(query")
normsrc = cell("def norm_id(s):", "def find_file_number")
tok = cell("def tokenize(t):", "bm25 = bm25s")

ns = dict(os=os, re=re, pd=pd, np=np, CHUNK_CHARS=900, CHUNK_OVERLAP=150)
exec(normsrc, ns)
long_text = "\n".join(f"line {i} fuel diesel invoice amount {i*10}" for i in range(80))
rows = []
def page(vn, pg, n, text, flags, conf=0.9):
    rows.append(dict(file_hash=vn.replace("/", "-").ljust(12, "0"), file_number=vn, doc_type=vn.split("/")[0], page=pg, n_pages=n, file_name=vn.replace("/", "-") + ".pdf",
                     rel="BE1/FE1/" + vn.replace("/", "-") + ".pdf", batch="BE1", folder="FE1", index_text=text, bm25_text=text,
                     text_source="ocr", page_flags=flags, ocr_mean_conf=conf))
page("IV/20/2223/0274", 1, 3, "Voucher IV/20/2223/0274 petty cash", [])
page("IV/20/2223/0274", 2, 3, "Garage bill Mercedes ABS sensor", ["low_conf_or_handwriting"], 0.55)
page("IV/20/2223/0274", 3, 3, long_text, ["table"])
page("STF/20/2223/0050", 1, 2, "Voucher STF/20/2223/0050 tour", [])
page("STF/20/2223/0050", 2, 2, "Toll receipt Lehra plaza", [])
ns["pages_df"] = pd.DataFrame(rows)
csv = os.path.join(__import__("tempfile").gettempdir(), "bc_test.csv")
open(csv, "w").write("Document_No,Vendor,Posting_Date,Amount,Department,Access_Tags\nIV-20-2223-0274,Acme Garage,2022-10-14,1250,LOG,dept:LOG\n")
ns["BC_CSV"] = csv
exec(sec9, ns)
chunks, parents = ns["chunks"], ns["parents"]

LONG_PID = "IV-20-2223-0274"[:12] + "#p3"   # parent_id = file_hash[:12] + "#p" + page
# 1 parent-child structure
assert len(parents) == 5, len(parents)
assert parents.at[LONG_PID, "n_children"] > 1, "long page must split into children"
assert (parents.n_children.drop(LONG_PID) == 1).all()
assert chunks.parent_id.isin(parents.index).all()
assert parents.at[LONG_PID, "text"] == long_text
# 2 metadata
assert set(parents.page_kind) == {"erp_voucher", "handwritten_or_faded", "printed_table", "printed"}
assert (chunks[chunks.voucher_no == "IV/20/2223/0274"].bc_vendor == "Acme Garage").all()
assert (chunks[chunks.voucher_no == "STF/20/2223/0050"].bc_match == "not_in_bc").all()
assert (chunks[chunks.voucher_no == "STF/20/2223/0050"].access_tags == "all").all()
assert "vendor Acme Garage" in chunks.embed_text.iat[0] and "date 2022-10-14" in chunks.embed_text.iat[0]
# 3 filters and access
allm = ns["visible_mask"]()
assert allm.all()
m = ns["visible_mask"](allowed_tags={"dept:FIN"})
assert set(chunks[m].voucher_no) == {"STF/20/2223/0050"}, "pages tagged dept:LOG must be hidden from a FIN-only user"
m = ns["visible_mask"](allowed_tags={"dept:LOG"})
assert m.all()
m = ns["visible_mask"](filters={"page_kind": ["handwritten_or_faded"]})
assert set(chunks[m].page_kind) == {"handwritten_or_faded"}
# 4 collapse
order = list(range(len(chunks)))
col = ns["collapse_to_parents"](order)
assert len(col) == len(set(chunks.parent_id.iloc[col])) == 5
# 5 search end to end with stubs
ns["tokenize"] = None; exec(tok, ns)
class BM:
    def retrieve(self, qtoks, k, show_progress, backend_selection):
        q = set(qtoks[0]); sc = np.array([len(q & set(ns["tokenize"](t))) for t in chunks.bm25_text], float)
        ids = np.argsort(-sc)[:k]; return np.array([ids]), np.array([sc[ids]])
class HN:
    def search(self, qv, k):
        q = set(ns["tokenize"](qv[0])); sc = np.array([len(q & set(ns["tokenize"](t))) for t in chunks.embed_text], float)
        return None, np.array([np.argsort(-sc)[:k]])
ns.update(bm25=BM(), hnsw=HN(), embed=lambda xs: list(xs), reranker=None, torch=types.SimpleNamespace(), norm_id=ns["norm_id"])
exec(srch, ns)
res = ns["search"]("fuel diesel invoice amount", k=5)
pids = [r["parent_id"] for r in res]
assert len(pids) == len(set(pids)), "each page once"
top = [r for r in res if r["parent_id"] == LONG_PID][0]
assert top["n_children"] > 1 and top["parent_text"] == long_text
res = ns["search"]("fuel diesel invoice amount", k=5, allowed_tags={"dept:FIN"})
assert all(r["parent_id"].startswith("STF") for r in res), res
res = ns["search"]("Mercedes ABS", k=3, file_filter="IV-20-2223-0274")
assert res and all(r["citation"].startswith("[IV/20/2223/0274") for r in res)
res = ns["search"]("Mercedes ABS", k=3, filters={"voucher_type": "STF"})
assert all(r["voucher_type"] == "STF" for r in res)
print("all checks passed;", len(chunks), "chunks,", len(parents), "parents")
