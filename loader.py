import os
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings
import re
from PyPDF2 import PdfReader
from openai import OpenAI
from utils import get_embedding
from tqdm import tqdm

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client with API key from environment variable
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize Chroma client : Persistent client to save this DB into the Disk. 
chroma_client = chromadb.PersistentClient(
    path="./chroma_db"
)

# Create a collection 
collection = chroma_client.create_collection("pdf_collection")

def extract_text_from_pdf(pdf_path):
    """
    Extract all the text from a PDF file.
    """
    with open(pdf_path, 'rb') as file:
        reader = PdfReader(file)
        text = ""
        # Iterate through all pages and extract text
        for page in reader.pages:
            text += page.extract_text()
    return text

def split_text_into_chunks(text, words_per_chunk=500, overlap=50):
    """
    Split text into smaller chunks for better embedding.
    """
    # Split text into words
    words = re.findall(r'\S+', text)
    chunks = []
    # Create overlapping chunks
    for i in range(0, len(words), words_per_chunk - overlap):
        chunk = ' '.join(words[i:i + words_per_chunk])
        chunks.append(chunk)
    return chunks

def process_pdfs(folder_path):
    """
    Process the PDFs: 
    1. Extract the text
    2. Convert the text into chunks
    3. Get embeddings for each chunk
    4. Load the chunks and the embeddings inside the chroma DB
    
    PS : We can also request OpenAI with batches...
    """

    doc_id = 0
    for filename in os.listdir(folder_path):
        if filename.endswith('.pdf'):
            print('file: ', filename)
            pdf_path = os.path.join(folder_path, filename)
            
            # Extract text from PDF
            print('Extract text from pdf')
            text = extract_text_from_pdf(pdf_path)
            
            # Split text into chunks
            print('Split text')
            chunks = split_text_into_chunks(text)
            print('Chunks size : ', len(chunks))
            for i, chunk in enumerate(chunks):
                # Create embeddings using OpenAI
                embedding = get_embedding(chunk)
                
                # Add to Chroma: Some of the VectorDB allow inserting batches
                collection.add(
                    embeddings=[embedding],
                    documents=[chunk],
                    metadatas=[{"source": filename, "chunk": i}],
                    ids=[f"{filename}_chunk_{i}"]
                )
                doc_id += 1

    print(f"Processed and added {len(collection.get()['ids'])} chunks to Chroma.")


def process_documents(documents):
    """
    If you want you can process a list of documents too: 
    1. Split the text into chunks
    2. Get embeddings for each chunk
    3. Load the chunks and the embeddings inside the chroma DB
    
    PS : We can also request OpenAI with batches...
    """
    import pickle
    from pathlib import Path

    # Load the toulouse_data_list.pickle from the specified folder
    data_path = Path("./data")
    data_file = data_path / "toulouse_data_list.pickle"
    with data_file.open('rb') as file:
        documents = pickle.load(file)

    doc_id = 0
    for document in tqdm(documents[:50]):
        # Split text into chunks
        chunks = split_text_into_chunks(document)
        for i, chunk in enumerate(chunks):
            # Create embeddings using OpenAI
            embedding = get_embedding(chunk)
            
            # Add to Chroma: Some of the VectorDB allow inserting batches
            collection.add(
                embeddings=[embedding],
                documents=[chunk],
                ids=[f"chunk_{i}"]
            )
            doc_id += 1

    print(f"Processed and added {len(collection.get()['ids'])} chunks to Chroma.")



# Usage
if __name__== '__main__':
    data_folder = "./data"  # Replace with your PDF folder path
    process_pdfs(data_folder)
