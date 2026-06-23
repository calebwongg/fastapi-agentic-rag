from rag import retrieve, flatten_context
import json 

def get_metrics(k: int): 
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
                break

    return (found / len(data["golden_set"])) * 100 

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
    max_percentage = 0
    for i in range(start, end): 
        percentage = trace_get_metrics(i) 
        print(f"the accuracy for k = {i} is {percentage}&")
        if percentage > max_percentage: 
            max_percentage = percentage
            max_index = i
    return max_index, max_percentage

if __name__ == '__main__':
    k, optimal_percentage = test_range(1, 20) 
    print(f'the most optimal k-index is {k} which has an accuracy of {optimal_percentage}%')
