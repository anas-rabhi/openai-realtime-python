from openai import OpenAI
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection("pdf_collection")
import chromadb

client = OpenAI()


def get_embedding(text):
    """
    Call OpenAI API to create embeddings for a given text.
    """
    # Call OpenAI API to generate embedding

    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

def query_chroma(query_embedding, n_results=5):
    """
    Query ChromaDB for relevant documents using the query embedding.
    """
    # Query ChromaDB
    collection = chroma_client.get_collection("pdf_collection")
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    # Return the first list of documents
    return results['documents'][0]



def rag_pipeline(query):
    """
    Executes the RAG pipeline.
    """
    
    # Generate embedding for the query
    question_embeddings = get_embedding(query)

    # Retrieve relevant chunks from ChromaDB
    relevant_chunks = query_chroma(question_embeddings, collection)

    return relevant_chunks

