import os
import sys
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from numpy.linalg import norm

def load_and_split_document(filepath, chunk_size_words=100, overlap_words=20):
    """Loads a text document and splits it into overlapping chunks."""
    if not os.path.exists(filepath):
        print(f"Error: File '{filepath}' not found.")
        print("Please create a text file with some content to test.")
        sys.exit(1)
        
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
        
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size_words - overlap_words):
        chunk_words = [words[j] for j in range(i, min(i + chunk_size_words, len(words)))]
        chunk = " ".join(chunk_words)
        if chunk.strip():
            chunks.append(chunk)
            
    return chunks

def cosine_similarity(A, B):
    """Computes cosine similarity between two vectors."""
    if norm(A) == 0 or norm(B) == 0:
        return 0.0
    return np.dot(A, B) / (norm(A) * norm(B))

def retrieve_top_k(query_embedding, chunk_embeddings, chunks, k=3):
    """Retrieves the top-k most similar chunks to the query."""
    similarities = [cosine_similarity(query_embedding, chunk_emb) for chunk_emb in chunk_embeddings]
    top_indices = np.argsort(similarities)[-k:][::-1]
    
    results = []
    for idx in top_indices:
        results.append({
            "chunk": chunks[idx],
            "score": similarities[idx]
        })
    return results

def generate_answer(user_query, retrieved_chunks_data, llm_pipeline):
    """Generates an answer using a local Hugging Face LLM."""
    
    # Format the retrieved chunks into a single context string
    retrieved_chunks = "\n\n".join([res['chunk'] for res in retrieved_chunks_data])
    
    # Exact prompt template from requirements
    prompt = f"""Answer the question based only on the context below.

Context:
{retrieved_chunks}

Question:
{user_query}

Answer:"""

    # Generate answer using the pipeline
    output = llm_pipeline(prompt, max_new_tokens=150, truncation=True)
    return output[0]['generated_text']

def main():
    # File to use
    filepath = "document.txt"
    
    # 1. Create a dummy file if it doesn't exist to ensure the script runs out of the box
    if not os.path.exists(filepath):
        print(f"Creating a sample file ({filepath}) for demonstration...")
        with open(filepath, "w", encoding='utf-8') as f:
            f.write("Retrieval-Augmented Generation (RAG) is a technique that enhances generative AI models with facts fetched from external sources. ")
            f.write("The primary advantage of RAG is that it grounds the model on truth, reducing hallucinations. ")
            f.write("Sentence Transformers are lightweight models used to convert text into vector embeddings. ")
            f.write("A vector embedding is an array of numbers representing the semantic meaning of text. ")
            f.write("Cosine similarity is an algorithm used to measure how similar two vectors are.")

    print("\n--- Starting Minimal RAG Pipeline ---")

    # 2. Load and Split Document
    print(f"Loading document: {filepath}")
    chunks = load_and_split_document(filepath)
    print(f"Document split into {len(chunks)} chunks.")

    # 3. Load Embedding Model
    print("Loading embedding model (sentence-transformers/all-MiniLM-L6-v2)...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    # 4. Create Vector Embeddings
    print("Generating embeddings for document chunks...")
    chunk_embeddings = embedding_model.encode(chunks)
    
    # 5. Load Language Model for Answer Generation (100% Local, No API Keys needed)
    print("Loading language model (google/flan-t5-small)... This may take a moment.")
    llm_pipeline = pipeline("text2text-generation", model="google/flan-t5-small")

    print("\n" + "="*50)
    print(" RAG System Ready (100% Local Edition)")
    print("="*50)
    print("Type your query below (or 'exit' to quit).\n")
    
    while True:
        try:
            user_query = input("Query > ")
            if user_query.strip().lower() in ['exit', 'quit']:
                break
            if not user_query.strip():
                continue
                
            # Convert query to embedding
            query_embedding = embedding_model.encode(user_query)
            
            # Retrieve top 3 relevant chunks
            k = min(3, len(chunks))
            top_results = retrieve_top_k(query_embedding, chunk_embeddings, chunks, k=k)
            
            # Print retrieved chunks for transparency (Optional Enhancement)
            print("\n--- Retrieved Relevant Chunks ---")
            for i, res in enumerate(top_results):
                print(f"[{i+1}] Score: {res['score']:.4f} | {res['chunk'][:100]}...")
                
            # Generate the final answer
            print("\nGenerating Answer...")
            answer = generate_answer(user_query, top_results, llm_pipeline)
            
            print("\n--- Final Answer ---")
            print(answer)
            print("-" * 50 + "\n")
                
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()
