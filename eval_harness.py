from rag import retrieve, flatten_context, build_prompt, generate
import json 
import ollama

#we are going to use mean reciprocal rank (mrr) for considering distance weights for each of our retrieved documents
# RR = 1 / rank (rank is how close it is to the query. technically not based on the numerical distance number but relative to all of the other documents fetched)
#the final mrr is the average of our scores across the data set
def get_metrics(k: int): 
    data = []
    mrr_sum = 0
    hits = 0
    hit_distances = []
    faithfulness_sum = 0 

    with open("golden_set.json", "r") as file: 
        data = json.load(file) #array of dictionaries

    for entry in data["golden_set"]:         
        raw_data = retrieve(entry["question"], k)
        documents = flatten_context(raw_data) #arr of dictionaries

        #logic for checking faithfulness 
        raw_data = retrieve(doc["question"], k)
        metadata = flatten_context(raw_data)
        query = build_prompt(doc["question"], metadata)
        result = generate(query)
        resp = get_faithfulness(query, result)
        faithfulness_sum += resp

        for rank, doc in enumerate(documents):
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
    avg_faithfulness = faithfulness_sum / len(data["golden_set"])
    
    return { 
        "hit_rate": hit_rate, 
        "mrr": avg_mrr, 
        "avg_hit_distance": avg_hit_distance,
        "avg_faithfulness": avg_faithfulness
    }


def get_faithfulness(prompt, response): 
    print(f'user prompt: {prompt}\n\n')
    print(f'llm response: {response}\n\n')
    print(f'checking faithfulness ...')
    query = f'''
        You are checking the faithfulness and evaluating the output of an AI assistant 
        Your task is to determine if the generated answer is entirely faithful to the provided retrieval context.
        Faithful means that the answer given by the assistant is fully and directely supported by the context provided in the original prompt.
        You are NOT fact checking the data based off of your prior background knowledge, you are only evaluating if the answer is supported by the context provided in the prompt.

        USER_QUERY: {prompt} 
        AI_ASSISTANT_RESPONSE: {response}
        
        Rules: 
        1. Break down the response into multiple factual claims
        2. Evaluate the AI assistant's response and based off of the broken down chunks, grade the response from 0-1.00 based off how faithful it is
        3. Return your faithfulness number evaluation in a float, and provide your reasoning for why you provided that score in the following json format: 
        {{
            "faithfulness": float 
            "reasoning": str
        }}
    '''
    resp = ollama.chat( 
        model = "llama3.2",
        messages = [{"role": "user", "content": query}],
        format = "json"
    )

    return resp


def test_range(start: int, end: int) -> tuple[int, int]:
    max_index = 0
    max_mrr = 0
    for i in range(start, end): 
        results = get_metrics(i) 
        print(f"the mrr for k = {i} is {results["mrr"]}")
        print(f'the hit rate for k = {i} is {results["hit_rate"]}')
        if results["mrr"] > max_mrr: 
            max_mrr = results["mrr"]
            max_index = i
    return max_index, max_mrr



if __name__ == '__main__':
    raw_data = retrieve("How do I define an optional query parameter with a default value?", 3)
    metadata = flatten_context(raw_data)
    query = build_prompt("How do I define an optional query parameter with a default value?", metadata)
    result = generate(query)
    raw_ollama_resp = get_faithfulness(query, result)
    json_string = raw_ollama_resp.message.content 
    resp = json.loads(json_string)
    print(f'raw response: {resp}')
    print(f'the faithfulness of this query is {resp["faithfulness"]}')
    print(f'ollamas reasoning is: {resp["reasoning"]}') 

    #k, optimal_mrr = test_range(1, 20) 
    #print(f'the most optimal k-index is {k} which has an mrr of {optimal_mrr}')
