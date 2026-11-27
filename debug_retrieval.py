import sys
from rag import retrieve

q = sys.argv[1] if len(sys.argv) > 1 else "Why do we use += instead of = in backprop?"
for c in retrieve(q, 6):
    print(f"{c['dist']:.2f} | {c['meta']['source']} ({c['meta']['location']})")
    print("    ", c['text'][:200].replace("\n", " "))