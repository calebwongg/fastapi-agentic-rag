import json
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from rag import retrieve, flatten_context, build_prompt, generate
import ollama

K = 4  

class RAGState(TypedDict):
    question: str      # original, never changes
    query: str         
    documents: list
    answer: str
    relevant: bool
    critique: str      # grader's note on WHY retrieval was weak
    attempts: int      # counts rewrites (loop guard)


def retrieve_node(state: RAGState):
    print('retrive')
    data = retrieve(state["query"], K)          # uses query, not question
    return {"documents": flatten_context(data)}

def grade_node(state: RAGState):                
    print('grade')
    docs = "\n\n".join(d["document"] for d in state["documents"])

    prompt = f"""Grade whether these documents are relevant enough to answer the question.
    Question: {state['question']}
    Documents: {docs}
    Return JSON only: {{"relevant": true or false, "critique": "if not relevant, what's missing or how to search better"}}"""

    resp = ollama.chat(model="llama3.2", messages=[{"role":"user","content":prompt}], format="json")

    try:
        v = json.loads(resp["message"]["content"])

    except json.JSONDecodeError:
        print('json decode error')
        v = {"relevant": True, "critique": ""}   # fail-open so we don't loop on bad JSON

    return {
        "relevant": v.get("relevant", False), 
        "critique": v.get("critique", "")
        }

def rewrite_node(state: RAGState):      
    print('rewrite')
    prompt = f"""Rewrite this into a better documentation search query.
    Original question: {state['question']}
    Why the last search was weak: {state.get('critique','')}
    Return ONLY the improved query, nothing else."""

    resp = ollama.chat(model="llama3.2", messages=[{"role":"user","content":prompt}])

    return {
        "query": resp["message"]["content"].strip(), 
        "attempts": state["attempts"] + 1
        }

def generate_node(state: RAGState):
    print('generate')
    answer = generate(build_prompt(state["question"], state["documents"]))
    return {
        "answer": answer
        }

def decide(state: RAGState) -> str:
    print('decide')
    if state["relevant"]:
        return "generate"
    if state["attempts"] >= 2:                    # give up rewriting and just answer with what we have
        return "generate"
    return "rewrite"

graph = StateGraph(RAGState)
graph.add_node("retrieve", retrieve_node)
graph.add_node("grade", grade_node)
graph.add_node("rewrite", rewrite_node)
graph.add_node("generate", generate_node)

graph.add_edge(START, "retrieve")
graph.add_edge("retrieve", "grade")
graph.add_conditional_edges("grade", decide, {"generate": "generate", "rewrite": "rewrite"})
graph.add_edge("rewrite", "retrieve")            # the corrective loop
graph.add_edge("generate", END)

app = graph.compile()

if __name__ == "__main__":
    q = "How do I declare a path parameter in FastAPI and give it a type?"  # q01, your showcase
    result = app.invoke({
        "question": q, "query": q, "documents": [],
        "answer": "", "relevant": False, "critique": "", "attempts": 0,
    })
    print(result["answer"])