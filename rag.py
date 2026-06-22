import chromadb
from chromadb.utils import embedding_functions

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

print(retrieve("how do i define a parameter in fastapi", 4)) 