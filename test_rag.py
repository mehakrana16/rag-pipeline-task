import csv, json, os, time
from rag import ask

CACHE = "test_cache.json"
cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}

TESTS = [
    ("What does the add gate do during backpropagation?", "equally", "Backpropagation.pdf"),
    ("What does the max gate do in the backward pass?", "higher", "Backpropagation.pdf"),
    ("In the example f = (x+y)z with x=-2, y=5, z=-4, what is f?", "-12", "Backpropagation.pdf"),
    ("Why do we use += instead of = in backprop?", "add", "Backpropagation.pdf"),
    ("How is dW computed for matrix multiplication?", "X.T", "Backpropagation.pdf"),
    ("What is the price of the Pro plan?", "499", "product_faq.md"),
    ("How often do I need to change my password?", "90", "it_security.md"),
    ("How long is the probation period?", "3 month", "onboarding.md"),
    ("How many casual leaves do I get?", "12", "leave_policy.md"),
    ("Can I work from home?", "2 days", "work_hours.md"),
    ("What is the meal limit during travel?", "800", "expenses.md"),
    ("Who is the CEO of Google?", None, None),
    ("Who won the 2022 World Cup?", None, None),
]
 
rows = []
for q, kw, src in TESTS:
    start = time.time()
    if q in cache:
        ans, sources = cache[q]["answer"], cache[q]["sources"]
    else:
        try:
            o = ask(q)
            ans, sources = o["answer"], o["sources"]
            cache[q] = {"answer": ans, "sources": sources}
            json.dump(cache, open(CACHE, "w"))
        except Exception as e:
            ans, sources = f"ERROR: {str(e)[:80]}", []
    secs = round(time.time() - start, 1)

    if kw:
        ok = kw.lower() in ans.lower() and any(src in s for s in sources)
    else:
        ok = "don't have enough information" in ans.lower() and not sources

    rows.append([q, kw or "refusal", ans, ", ".join(sources) or "none", "PASS" if ok else "FAIL", secs])
    print(f"{'PASS' if ok else 'FAIL'} | {q} -> {ans} | {sources} | {secs}s")

print(f"\nResult: {sum(r[4] == 'PASS' for r in rows)}/{len(rows)} passed")
with open("test_results.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["Question", "Expected", "Actual answer", "Source", "Result", "Seconds"])
    w.writerows(rows)
