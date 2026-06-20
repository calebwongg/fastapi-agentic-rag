from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document 

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

content = read_docs() 
chunks = create_chunks(content)
print("\n sample chunk")
print(f"source file: {chunks[5].metadata['source']}") #.metadata['source'] attribute of langchain Document object
print(f"content: {chunks[5].page_content}")
