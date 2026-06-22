import chromadb
from chromadb.utils import embedding_functions
import ollama

#ollama pull downloads the specific model onto your local machien. ollama pull 3.2 is necessary for running and utilizing the ollama model

def retrieve(prompt: str, k: int): 
    chroma_client = chromadb.PersistentClient(path="./chromadb") #./ specifies current directory
    local_ef = embedding_functions.DefaultEmbeddingFunction()
    collection = chroma_client.get_collection( 
        name = "fastapi_docs", 
        embedding_function = local_ef,
    )
    #the documents are the actual documents we loaded into the vector embed. the data that we are grabbing. the distances are the 
    #lengths representing how far the document is from the query in the embed / how accurate the document is. lower distance = more accurate
    #in the returned results, documents and distances are parallely related, theres not necessarily a direct connection between them in the dictionary
    results = collection.query(query_texts=[prompt], n_results=k, include=["documents", "distances"])
    '''
    the shape of results looks like this 
    results = {
    "ids": [
        ["id1", "id2", "id3", "id4"]
    ],
    "distances": [
        [0.12, 0.24, 0.35, 0.48]  # 4 floats sorted from closest to furthest
    ],
    "documents": [
        ["text1", "text2", "text3", "text4"]  # 4 text strings
    ],
    "metadatas": [
        [{"source": "pdf"}, {"source": "web"}, {"source": "pdf"}, {"source": "txt"}]
    ],
    "embeddings": None  # returns none unelss we ask for embeddings. this are the actual embeddings / the long array of numbers numerically representing the documents
}
    '''

    return results


def generate(prompt, context): 

    query = f"""
        Act as a documentation-backed FastAPI Agentic Assistant tool. 
        
        Return answers and guidance based off of the user prompt and the documents provided. 
        Use ONLY the provided retrieved context to answer the user's question. Cite the sources that you received 
        your answers from to ensure credibility for your responses. Do not use any external knowledge

        Rules:
        1. Be polite, concise, and professional.
        2. If the answer cannot be found in the provided context, reply exactly with: "I'm sorry, but I cannot find that information in our records."
        3. Cite the source document name if provided in the context.

        USER_QUESTION: {prompt}
        CONTEXT: {context}
    """

    resp = ollama.chat( 
        model = "llama3.2",
        messages=[{"role": "user", "content": prompt}], 
    )

    answer = resp["message"]["content"]
    return answer


prompt = "how do i define a paramter in fastapi"
k = 4
documents = retrieve(prompt, k) 
print(f'documents: {documents}')
answer = generate(prompt, documents)
print(answer) 