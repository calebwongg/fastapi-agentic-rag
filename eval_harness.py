from rag import retrieve, flatten_context
import json 

#we are going to use mean reciprocal rank (mrr) for considering distance weights for each of our retrieved documents
# RR = 1 / rank (rank is how close it is to the query. technically not based on the numerical distance number but relative to all of the other documents fetched)
#the final mrr is the average of our scores across the data set
def get_metrics(k: int): 
    data = []
    mrr_sum = 0
    hits = 0
    hit_distances = []

    with open("golden_set.json", "r") as file: 
        data = json.load(file) #array of dictionaries

    for entry in data["golden_set"]: 
        raw_data = retrieve(entry["question"], k)
        documents = flatten_context(raw_data) #arr of dictionaries
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
        avg_hit_distance = sum(hit_distances) / len(hit_distances])
    else: 
        avg_hit_distance = 0
    
    return { 
        "hit_rate": hit_rate, 
        "mrr": avg_mrr, 
        "avg_hit_distance": avg_hit_distance
    }


def trace_get_metrics(k: int): 
    data = []
    found = 0
    with open("golden_set.json", "r") as file: 
        data = json.load(file) #array of dictionaries

    for entry in data["golden_set"]: 
        raw_data = retrieve(entry["question"], k)
        documents = flatten_context(raw_data) #arr of dictionaries
        for doc in documents: 
            if doc["source"] in entry["expected_sources"]: 
                found += 1
                print(f'confirmed: found {doc["source"]} in {entry["id"]}')
                break

    print(f'found: {found}. number of test case: {len(data["golden_set"])}')
    return (found / len(data["golden_set"])) * 100 




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
    k, optimal_mrr = test_range(1, 20) 
    print(f'the most optimal k-index is {k} which has an mrr of {optimal_mrr}')
