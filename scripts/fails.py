"""Print the questions the last eval run got wrong (reads results/per_question.jsonl)."""
import json, sys
qs = {}
for l in open("eval/public_eval.jsonl"):
    d = json.loads(l); qs[d["id"]] = d
thr = float(sys.argv[1]) if len(sys.argv) > 1 else 0.99
for l in open("results/per_question.jsonl"):
    r = json.loads(l)
    if r["score"] < thr:
        a = r["answer"]; q = qs[r["id"]]
        print(f"\n{r['id']} [{r['category']}] {r['score']:.2f} trap={r.get('trap_hits')} miss={r.get('missed')}\n Q: {q['question']}\n A: {a['text'][:260]}\n src={[s['url'].replace('https://www.dupaco.com','')[:60] for s in a.get('sources',[])][:3]}")
