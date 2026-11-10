"""Load the snapshots into the solution and run ad-hoc questions: python scripts/dbg.py "question" [as_of]"""
import json, sys, glob, os
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(root, "harness")); sys.path.insert(0, os.path.join(root, "solution"))
from contract import Page, Snapshot
import solution
def load():
    ks = solution.build()
    for d in sorted(glob.glob(os.path.join(root, "snapshots", "0*"))):
        m = json.load(open(d + "/manifest.json"))
        pages = [Page(p["url"], p["title"], open(os.path.join(d, p["file"])).read(), p["fetched_at"]) for p in m["pages"]]
        ks.ingest(Snapshot(m["snapshot_id"], m["taken_on"], pages))
    return ks
if __name__ == "__main__":
    ks = load()
    for q in sys.argv[1:]:
        a = ks.answer(q)
        print("Q:", q, "\nA:", a.text[:600], "\nconf:", a.confident, [s.url for s in a.sources][:3], "\n")
