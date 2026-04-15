import rag_pipeline
import warnings
warnings.filterwarnings('ignore')

print("1. Loading Document...")
chunks = rag_pipeline.load_and_split_document("document.txt")

print("2. Loading Embedding Model... (all-MiniLM-L6-v2)")
from sentence_transformers import SentenceTransformer
emb_model = SentenceTransformer('all-MiniLM-L6-v2')

print("3. Vectorizing Chunks...")
chunk_embeddings = emb_model.encode(chunks)

print("4. Loading Language Model... (google/flan-t5-small)")
from transformers import pipeline
llm = pipeline("text2text-generation", model="google/flan-t5-small")

query = "What is the primary advantage of RAG?"
print(f"\nQuery: {query}")

query_emb = emb_model.encode(query)
top_results = rag_pipeline.retrieve_top_k(query_emb, chunk_embeddings, chunks, k=3)

print("5. Generating Final Answer...")
answer = rag_pipeline.generate_answer(query, top_results, llm)

print("\n" + "="*50)
print(f"Top Retrieved Context: {top_results[0]['chunk'][:100]}...")
print("="*50)
print(f"Final Answer: {answer}")
print("="*50)
