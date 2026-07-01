import langgraph
from typing import TypedDict
from rag import retrieve, flatten_context, build_prompt, generate

class RAGState(TypedDict):
    question: str        # original user input
    documents: list       # retrieved + flattened chunks
    query: str           # rebuilt user prompt with relevant documents. result from build_prompt
    answer: str          # final generated answer
    relevant: bool       # grade node's verdict
    attempts: int        # loop guard that represents how many rewrites / self correction iterations we have

def set_prompt(prompt: str, state: RAGState): 
    return { 
        "question": prompt
    }

def retrieve_node(k: int, state: RAGState):
    question = state["question"]
    data = retrieve(question, k)
    documents = flatten_context(data)
    return { 
        "documents": documents
    }

def build_prompt_node(state: RAGState): 
    built_prompt = build_prompt(state["question"], state["documents"])
    return { 
        "query": built_prompt
    }

def generate_node(state: RAGState): 
    answer = generate(state["query"])
    return { 
        "answer": answer
    }






'''
if __name__ == '__main__':
    prompt = "How do I split a large FastAPI app across multiple files?"
    k = 4
    documents = retrieve(prompt, k) 
    data = flatten_context(documents)
    print(f'documents: {data}')

'''