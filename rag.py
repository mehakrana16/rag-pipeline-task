import os, glob
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from pydantic import BaseModel
from dotenv import load_dotenv
from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception
 
load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
MODELS = ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-3.7-flash", "gemini-3.8-flash"]
embedder = SentenceTransformer("all-MiniLM-L6-v2")
db = chromadb.PersistentClient(path="chroma_db")


def load_docs(folder="docs"):
    for path in glob.glob(f"{folder}/*"):
        name = os.path.basename(path)
        if path.endswith(".pdf"):
            for i, page in enumerate(PdfReader(path).pages, 1):
                yield name, f"page {i}", page.extract_text() or ""
        elif path.endswith((".md", ".txt")):
            yield name, "full document", open(path, encoding="utf-8").read()


def build_index(chunk_size=800, overlap=120):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    try:
        db.delete_collection("docs")
    except Exception:
        pass
    col = db.get_or_create_collection("docs", metadata={"hnsw:space": "cosine"})
    ids, texts, metas = [], [], []
    for name, loc, text in load_docs():
        for j, ch in enumerate(splitter.split_text(text)):
            ids.append(f"{name}-{loc}-{j}")
            texts.append(ch)
            metas.append({"source": name, "location": loc})
    col.add(ids=ids, documents=texts, metadatas=metas,
            embeddings=embedder.encode(texts).tolist())
    print(f"Indexed {len(texts)} chunks")


def retrieve(q, k=4):
    col = db.get_collection("docs")
    r = col.query(query_embeddings=embedder.encode([q]).tolist(), n_results=k)
    return [{"text": d, "meta": m, "dist": s}
            for d, m, s in zip(r["documents"][0], r["metadatas"][0], r["distances"][0])]


class Answer(BaseModel):
    answer: str
    sources: list[str]
    found: bool


SYSTEM = """Answer ONLY using the provided context. Cite sources as 'filename (location)'.
If the context does not contain the answer, set found=false, sources=[] and answer exactly:
"I don't have enough information." Never use outside knowledge."""


def _retryable(e):
    msg = str(e)
    return not ("429" in msg or "RESOURCE_EXHAUSTED" in msg)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=3, max=15),
       retry=retry_if_exception(_retryable), reraise=True)
def _call(model, contents):
    return client.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM,
            response_mime_type="application/json",
            response_schema=Answer,
            temperature=0,
            http_options=types.HttpOptions(timeout=30000),
        ),
    )


def call_llm(contents):
    last = None
    for m in MODELS:
        try:
            return _call(m, contents)
        except Exception as e:
            print(f"[{m} failed: {str(e)[:60]}] trying next model...")
            last = e
    raise last


def ask(q, k=4, max_dist=0.65):
    chunks = [c for c in retrieve(q, k) if c["dist"] <= max_dist]
    if not chunks:
        return {"answer": "I don't have enough information.", "sources": [], "found": False}
    ctx = "\n\n".join(f"[{c['meta']['source']} ({c['meta']['location']})]\n{c['text']}" for c in chunks)
    r = call_llm(f"Context:\n{ctx}\n\nQuestion: {q}")
    return Answer.model_validate_json(r.text).model_dump()


if __name__ == "__main__":
    import sys
    if "--rebuild" in sys.argv or not os.path.exists("chroma_db"):
        build_index()
    print("Ask a question (type 'exit' to quit)")
    while True:
        q = input("\nQ: ").strip()
        if q.lower() in ("exit", "quit"):
            break
        o = ask(q)
        print(f"A: {o['answer']}\nSources: {o['sources']}")
