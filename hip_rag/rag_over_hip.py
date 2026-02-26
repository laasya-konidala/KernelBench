import git
from firecrawl import Firecrawl
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
import chromadb
import numpy as np
from pathlib import Path
import json
from dotenv import load_dotenv
import os
load_dotenv()


FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")

HIP_REPOS = {
    "HIP":          "https://github.com/ROCm/HIP",           # core HIP
    "ROCm":         "https://github.com/ROCm/ROCm",           # main ROCm docs
    "rocm-examples": "https://github.com/ROCm/rocm-examples", # HIP examples + tutorials
}

CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
FIRECRAWL_LIMIT = 1
FIRECRAWL_URL   = "https://rocm.docs.amd.com/projects/HIP/en/latest/"
DB_PATH         = "./hip_rag_db"
COLLECTION_NAME = "hip_docs"
TOP_K           = 10 # number of chunks to retrieve per query

# Clone the HIP, ROCm, and rocm-examples repos from GitHub
# Just clone the latest commit and skip if already has been cloned
def clone_repos():
    for name, url in HIP_REPOS.items():
        destination = f"./{name}"
        if Path(destination).exists():
            print(f"'{name}' already exists, skipping.")
        else:
            print(f"Cloning {name}...")
            git.Repo.clone_from(url, destination, depth=1)
            print(f"Done cloning: {name}")
    for name in HIP_REPOS:
        readme_files = list(Path(f"./{name}").rglob("*.md"))
        print(f"{name}: {len(readme_files)} .md files")
        
# Use Firecrawl to scrape (which will convert each page to a markdown)
# Save the results in firecrawl_cache.json so you only need to scrape once
# because I only have 500 free scrapes
def scrape_using_firecrawl():
    cache_path = Path("./firecrawl_cache.json")

    if cache_path.exists():
        print("Loading scraped pages from cache...")
        with open(cache_path, "r") as f:
            data = json.load(f)
        print(f"Loaded {len(data)} pages from cache.")
        return data

    from firecrawl import Firecrawl
    app = Firecrawl(api_key=FIRECRAWL_API_KEY)
    print(f"Crawling {FIRECRAWL_URL} (limit={FIRECRAWL_LIMIT})...")
    result = app.crawl(
        FIRECRAWL_URL,
        limit=FIRECRAWL_LIMIT,
        scrape_options={"formats": ["markdown"]}
    )
    pages = result.data

    cache_data = [
        {
            "markdown": page.markdown,
            "source_url": page.metadata.source_url if page.metadata else "rocm_docs"
        }
        for page in pages if page.markdown
    ]

    with open(cache_path, "w") as f:
        json.dump(cache_data, f)
    print(f"Scraped {len(pages)} pages. Saved to {cache_path}.")
    return cache_data
    
def load_and_chunk(pages):
    # Load the GitHub markdown files 
    git_docs = []
    for repo_name in HIP_REPOS:
        loader = DirectoryLoader(f"./{repo_name}",
                                 glob="**/*.md",
                                 loader_cls=TextLoader,
                                 loader_kwargs={"encoding": "utf-8"},
                                 show_progress=True,
                                 silent_errors=True, # if a file fails to load just skip it 
                                 )
        docs = loader.load()
        print(f"{repo_name}: {len(docs)} docs")
        git_docs.extend(docs)
        
    # Convert the Firecrawl documents to LangChain documents
    firecrawl_docs = []
    for page in pages:
        if isinstance(page, dict):
            markdown = page.get("markdown", "")
            source_url = page.get("source_url", "rocm_docs")
        else:
            markdown = page.markdown
            source_url = page.metadata.source_url if page.metadata else "rocm_docs"
        if markdown:
            firecrawl_docs.append(Document(page_content=markdown,
                                           metadata={"source": source_url}
                                           ))
    print(f"Firecrawl: {len(firecrawl_docs)} docs")
    
    # Combine all the documents (GitHub and Firecrawl)
    all_docs = git_docs + firecrawl_docs
    print(f"Total docs: {len(all_docs)}")
    
    # Split on markdown headers
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
        strip_headers=False, 
    )
    
    # Recursively split the text with separators 
    char_splitter = RecursiveCharacterTextSplitter(chunk_size = CHUNK_SIZE,
                                                   chunk_overlap=CHUNK_OVERLAP,
                                                   separators=["\n```\n", "\n\n", "\n", " ", ""],
                                                   )
    
    all_chunks = []
    for doc in all_docs:
        # Split by headers
        header_splits = header_splitter.split_text(doc.page_content)
        # Keep track of what file each chunk came from 
        for header_chunk in header_splits:
            header_chunk.metadata.update(doc.metadata)
        # Also split by character
        all_chunks.extend(char_splitter.split_documents(header_splits))
        
    print(f"Total chunks: {len(all_chunks)}")
    
    return all_chunks

def embed(all_chunks):
    print("Now loading the embedding model...")
    embedder = SentenceTransformer("BAAI/bge-large-en-v1.5")
    texts = [chunk.page_content for chunk in all_chunks]
    print(f"Embedding {len(texts)} chunks...")
    embeddings = embedder.encode(texts, batch_size = 32,
                                 normalize_embeddings=True,
                                 show_progress_bar=True)
    print(f"Done. Shape: {embeddings.shape}")
    
    return embedder, texts, embeddings

def build_index(all_chunks, texts, embeddings):
    client = chromadb.PersistentClient(path=DB_PATH)
    try:
        client.delete_collection(COLLECTION_NAME)
        print("Deleted old collection.")
    except:
        pass
    
    collection = client.get_or_create_collection(name=COLLECTION_NAME,
                                                 metadata={"hnsw:space": "cosine"})
    
    for i in range(0, len(texts), 500):
        batch_texts = texts[i:i+500]
        batch_embs = embeddings[i:i+500].tolist()
        batch_ids = [str(j) for j in range(i, i + len(batch_texts))]
        batch_metas = [ {k: str(v) for k, v in chunk.metadata.items() if v is not None} for chunk in all_chunks[i : i + 500] ]
        collection.upsert(
        documents=batch_texts,
        embeddings=batch_embs,
        ids=batch_ids,
        metadatas=batch_metas,
        )
        print(f"Indexed {min(i + 500, len(texts))}/{len(texts)}")

    print(f"\nDone! {collection.count()} chunks indexed.")
    return collection

def load_index():
    return chromadb.PersistentClient(path=DB_PATH).get_collection(COLLECTION_NAME)

def load_embedder():
    return SentenceTransformer("BAAI/bge-large-en-v1.5")

def retrieve(query, collection, embedder):
    q_emb = embedder.encode([query], normalize_embeddings=True).tolist()
    results = collection.query(query_embeddings=q_emb, n_results=TOP_K)
    return results["documents"][0], results["metadatas"][0]

def get_rag_context(query: str, collection=None, embedder=None) -> str:
    if collection is None:
        collection = load_index()
    if embedder is None:
        embedder = load_embedder()
    docs, metas = retrieve(query, collection, embedder)
    context_parts = []
    for i in range(len(docs)):
        doc = docs[i]
        meta = metas[i]
        source = meta.get("source", "unknown")
        label = Path(source).name
        if source.startswith("http"):
            label = source
        header = " > ".join(filter(None, [meta.get("h1"), meta.get("h2"), meta.get("h3")]))
        label  = f"[{label}]" + (f" {header}" if header else "")
        context_parts.append(f"{label}\n{doc}")
    return "\n\n---\n\n".join(context_parts)

if __name__ == "__main__":
    clone_repos()
    # pages = scrape_using_firecrawl()
    pages = []
    all_chunks = load_and_chunk(pages)
    embedder, texts, embeddings = embed(all_chunks)
    build_index(all_chunks, texts, embeddings)
    print("\n DONE!")