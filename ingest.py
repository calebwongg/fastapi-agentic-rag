from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document 
from google import genai
from google.genai import types 
from dotenv import load_dotenv
import chromadb
import os
from chromadb.utils.embedding_functions import GoogleGeminiEmbeddingFunction
import time
from chromadb.utils import embedding_functions

dir_path = Path("data")
def read_docs() -> list[dict[str, str]]: 
    total_characters = 0
    docs: list[dict[str,str]] = []
    for file in dir_path.rglob("*.md"): #only pull from the md files. the rglob handles going thru every nested subfolder recursively    
        if file.is_file(): 
            content = ''

            with open(file, "r", encoding = 'utf-8') as my_file: 
                content = my_file.read() #one big string

            docs.append({
                "source": str(file.relative_to(dir_path)), 
                "content" : content
                })
            
            total_characters += len(content)
    print(f"Loaded {total_characters} characters")
    print(f"Processed {len(docs)} documents")
    return docs
    

#the rule of thumb for chunking overlap is 10%-15% of your chunk size
def create_chunks(content): 
    splitter = RecursiveCharacterTextSplitter ( 
        chunk_size = 1000, #about one thought? just trying this for now 
        chunk_overlap = 150, #15% of our chunk size
        length_function = len,
        separators=["\n\n", "\n", " ", ""] #prevent from splitting mid sentence
    )

    input_docs = [Document(page_content = doc["content"], metadata = {"source": doc["source"]}) for doc in content]

    #we are essentially just abstracting/breaking down individual documents into smaller pieces here (js chunking)
    #creates chunked langchain Document objects
    langchain_docs = splitter.split_documents(input_docs) #the splitter here is smart enough to not overlap across files

    print(f'Created {len(langchain_docs)} document objects / chunks')
    return langchain_docs


def embed(langchain_docs): 
    load_dotenv()

    '''
    client = genai.Client() 
    chroma_client = chromadb.PersistentClient(path = "./chromadb")
    print("active key:", os.getenv("GEMINI_API_KEY")[:8])
    somethig is wrong with gcp allowing my gemini key. apparentely its a widespread current bug
    '''
    chroma_client = chromadb.PersistentClient(path="./chromadb") #wriet to disk
    local_ef = embedding_functions.DefaultEmbeddingFunction()  # runs locally now
    collection = chroma_client.get_or_create_collection(
        name="fastapi_docs",          
        embedding_function=local_ef,
    )

    batch_size = 100
    for i in range(0, len(langchain_docs), batch_size):
        batch = langchain_docs[i:i + batch_size]
        print(f"embedding batch {i // batch_size + 1}: chunks {i} to {i + len(batch)}...")
        collection.upsert(
            ids=[f"chunk_{j}" for j in range(i, i + len(batch))],
            documents=[doc.page_content for doc in batch],  
            metadatas=[doc.metadata for doc in batch],
        )
    time.sleep(2) #for rate limits

    print(f"done {collection.count()} vectors in collection.")
    return collection
    




content = read_docs() 
chunks = create_chunks(content)
print("\n sample chunk")
print(f"source file: {chunks[5].metadata['source']}") #.metadata['source'] attribute of langchain Document object
print(f"content: {chunks[5].page_content}")

embed(chunks)