import json, collections, sys

path = sys.argv[1] if len(sys.argv) > 1 else "/root/autodl-tmp/raw/xes3g5m/metadata/questions.json"
d = json.load(open(path, encoding="utf-8"))
ks = list(d.keys())
print("total questions:", len(ks))

types = collections.Counter(str(d[k].get("type")) for k in ks)
print("type counts:", dict(types))

# how many have a non-empty options field, and what python type it is
opt_type = collections.Counter(type(d[k].get("options")).__name__ for k in ks)
print("options field python-type counts:", dict(opt_type))

with_list = [k for k in ks if isinstance(d[k].get("options"), list) and d[k].get("options")]
with_any = [k for k in ks if d[k].get("options")]
print("with list-options:", len(with_list), " with any truthy options:", len(with_any))

# show a couple of samples for whichever branch exists
sample_keys = with_list[:2] if with_list else (with_any[:2] if with_any else ks[:2])
for k in sample_keys:
    q = d[k]
    print("\n--- qid", k, "type", q.get("type"), "---")
    print("keys:", list(q.keys()))
    print("options repr:", repr(q.get("options"))[:400])
    print("answer repr:", repr(q.get("answer"))[:120])
