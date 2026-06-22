import chromadb


client = chromadb.PersistentClient(path="./chromadb")
collection = client.get_collection(name="fastapi_docs")
print(f"total chunks in database: {collection.count()}")