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
    docs_list = state["documents"]          

    #try to prevent any over correction and regressions. deterministic grading laying
    if docs_list and docs_list[0]["distance"] < 0.6:
        print('  -> strong match, skipping grade')
        return {"relevant": True, "critique": ""}

    docs = "\n\n".join(d["document"] for d in docs_list)   
    prompt = f"""You are grading whether retrieved documents can answer a question.
        Default to RELEVANT. Only mark irrelevant if the documents are CLEARLY off-topic —
        ...
        Documents: {docs}
        Return JSON only: {{"relevant": true or false, "critique": "only if irrelevant, what's missing"}}"""
    resp = ollama.chat(model="llama3.2", messages=[{"role":"user","content":prompt}], format="json")

    try:
        v = json.loads(resp["message"]["content"])
    except json.JSONDecodeError:
        print('json decode error')
        v = {"relevant": True, "critique": ""}

    return {"relevant": v.get("relevant", True), "critique": v.get("critique", "")}

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


def get_metrics_orchestrated(k: int): 
    data = []
    mrr_sum = 0
    hits = 0
    hit_distances = []

    with open("golden_set.json", "r") as file: 
        data = json.load(file) #array of dictionaries

    for entry in data["golden_set"]:
        q = entry["question"]   

        result = app.invoke({
        "question": q, "query": q, "documents": [],
        "answer": "", "relevant": False, "critique": "", "attempts": 0,
        })
        
        for rank, doc in enumerate(result["documents"]):
            if doc["source"] in entry["expected_sources"]: 
                hits += 1
                mrr_sum += (1.0 / (rank + 1)) #indexed starting at 0 so add 1
                hit_distances.append(doc["distance"])
                break #for this logic right now, we are only looking at the first matched doc. this algorithm can/should improve later on

    hit_rate = hits / len(data["golden_set"])
    avg_mrr = (mrr_sum / len(data["golden_set"]))
    avg_hit_distance = 0 
    if hit_distances: 
        avg_hit_distance = sum(hit_distances) / len(hit_distances)
    else: 
        avg_hit_distance = 0
    
    return { 
        "hit_rate": hit_rate, 
        "mrr": avg_mrr, 
        "avg_hit_distance": avg_hit_distance,
    }

def average_faithfulness(k): 
    faithfulness_sum = 0 
    reasonings = []
    with open("golden_set.json", "r") as file: 
        data = json.load(file)

    for entry in data["golden_set"]:
        q = entry["question"]
        result = app.invoke({
            "question": q, "query": q, "documents": [],
            "answer": "", "relevant": False, "critique": "", "attempts": 0,
        })
        context = result["documents"]     # the agent's FINAL retrieved docs
        answer = result["answer"]         # the agent's answer
        raw = get_faithfulness_orchestrated(context, answer)
        try:
            resp = json.loads(raw.message.content)
        except json.JSONDecodeError:
            resp = {"faithfulness": 0.0, "reasoning": "FAILED_TO_PARSE"}
        faithfulness_sum += resp["faithfulness"]
        reasonings.append({"entry": q, "faithfulness": resp["faithfulness"], "reasoning": resp["reasoning"]})

    return faithfulness_sum / len(data["golden_set"]), reasonings

def get_faithfulness_orchestrated(context, response): 
    #print(f'user prompt: {prompt}\n\n')
    #print(f'llm response: {response}\n\n')
    print(f'checking faithfulness ...')
    query = f'''
        You are checking the faithfulness and evaluating the output of an AI assistant 
        Your task is to determine if the generated answer is entirely faithful to the provided retrieval context.
        Faithful means that the answer given by the assistant is fully and directely supported by the context provided in the original prompt.
        You are NOT fact checking the data based off of your prior background knowledge, you are only evaluating if the answer is supported by the context provided in the prompt.

        CONTEXT: {context} 
        AI_ASSISTANT_RESPONSE: {response}
        
        Rules: 
        1. Break down the response into multiple factual claims
        2. Evaluate the AI assistant's response and based off of the broken down chunks, grade the response from 0-1.00 based off how faithful it is
        3. Return your faithfulness number evaluation in a float, and provide your reasoning for why you provided that score in the following json format: 
        {{
            "faithfulness": float,
            "reasoning": str
        }}
    '''
    resp = ollama.chat( 
        model = "llama3.2",
        messages = [{"role": "user", "content": query}],
        format = "json"
    )

    return resp

if __name__ == "__main__":
    metrics = get_metrics_orchestrated(K)
    print(f"AGENT hit rate: {metrics['hit_rate']}")
    print(f"AGENT mrr: {metrics['mrr']}")
    avg_faith, reasonings = average_faithfulness(K)
    print(f"AGENT faithfulness: {avg_faith}")
    for r in reasonings:
        print(r)

