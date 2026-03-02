# EcoHome Energy Advisor - RAG Setup
# Sets up the Retrieval-Augmented Generation (RAG) pipeline so the agent can
# retrieve and cite relevant energy-saving tips and best practices.
#
# All .txt files in data/documents/ are loaded automatically.
# Add new documents there and re-run this script to include them.

# =============================================================================
# 1. Imports
# =============================================================================
import os
import sys
import shutil
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

_SRC_DIR = Path(__file__).resolve().parent
_DATA_DIR = _SRC_DIR / "data"
sys.path.insert(0, str(_SRC_DIR))

load_dotenv()

# =============================================================================
# 2. Load and Process Documents
# =============================================================================
# Load all .txt files from data/documents/ automatically.
documents = []
doc_dir = str(_DATA_DIR / "documents")

txt_files = sorted(f for f in os.listdir(doc_dir) if f.endswith(".txt"))

for filename in txt_files:
    doc_path = os.path.join(doc_dir, filename)
    loader = TextLoader(doc_path)
    docs = loader.load()
    documents.extend(docs)
    print(f"Loaded {len(docs)} document(s) from {doc_path}")

print(f"\nTotal documents loaded: {len(documents)}")

# =============================================================================
# 3. Split Documents into Chunks
# =============================================================================
# chunk_size=1000 chars with 200-char overlap balances retrieval precision and
# context. Adjust chunk_size (e.g. 500, 1500) to tune performance.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    separators=["\n\n", "\n", " ", ""],
)

splits = text_splitter.split_documents(documents)
print(f"Split {len(documents)} documents into {len(splits)} chunks")

if splits:
    print(f"\nSample chunk (first 200 characters):")
    print(splits[0].page_content[:200] + "...")

# =============================================================================
# 4. Create Vector Store
# =============================================================================
persist_directory = str(_DATA_DIR / "vectorstore")

# Fully remove any previous vectorstore so ChromaDB starts clean.
# (Chroma 0.6+ can fail with "readonly database" if stale WAL/journal files exist.)
if os.path.exists(persist_directory):
    shutil.rmtree(persist_directory)
os.makedirs(persist_directory)

embeddings = OpenAIEmbeddings(
    base_url="https://openai.vocareum.com/v1",
    api_key=os.getenv("OPENAI_API_KEY"),  # type: ignore
)

# PersistentClient ensures the SQLite file is created with correct permissions.
chroma_client = chromadb.PersistentClient(path=persist_directory)

vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    client=chroma_client,
)

print(f"\nVector store created and persisted to {persist_directory}")
print(f"Total vectors stored: {len(splits)}")

# =============================================================================
# 5. Test the RAG Pipeline (direct similarity search)
# =============================================================================
test_queries = [
    "electric vehicle charging tips",
    "thermostat optimization",
    "dishwasher energy saving",
    "solar power maximization",
    "HVAC system efficiency",
    "pool pump scheduling",
]

print("\n=== Testing Vector Search ===")
for query in test_queries:
    print(f"\nQuery: '{query}'")
    docs = vectorstore.similarity_search(query, k=2)
    for i, doc in enumerate(docs):
        print(f"  Result {i+1}: {doc.page_content[:100]}...")

# =============================================================================
# 6. Test the search_energy_tips Tool
# =============================================================================
from tools import search_energy_tips

test_queries = [
    "electric vehicle charging",
    "thermostat settings",
    "dishwasher optimization",
    "solar power tips",
]

print("\n=== Testing search_energy_tips Tool ===")
for query in test_queries:
    print(f"\nQuery: '{query}'")
    result = search_energy_tips.invoke({"query": query, "max_results": 3})

    if "error" in result:
        print(f"  Error: {result['error']}")
    else:
        print(f"  Found {result['total_results']} results")
        for i, tip in enumerate(result["tips"]):
            print(f"    {i+1}. {tip['content'][:100]}...")
            print(f"       Source: {tip['source']}")
            print(f"       Relevance: {tip['relevance_score']}")
