from pathlib import Path


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
    

                

read_docs()