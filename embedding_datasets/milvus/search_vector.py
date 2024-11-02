import json
import time
from embedding_datasets.milvus.create_collection import get_client, collection_name
from embedding_datasets.milvus.insert_vector import from_word_to_embeddings


def test(text):
    client = get_client()
    start = time.time()
    # text = "o-Isopropoxyphenyl N-methylcarbamate"
    print("-------start------", start)
    res = client.search(collection_name=collection_name, data=[from_word_to_embeddings(text)["average_embedding"]], limit=10,
                        search_params={
                            "metric_type": "COSINE"
                        }, output_fields=["average_embedding", "cmpdname", "origin_text"])
    print("-------start------", time.time() - start)
    result = [i["entity"]["origin_text"] for i in res[0]]
    print(result)


def search_by_bulk_vector(vector_list, limit=10, topk=10):
    client = get_client()
    res = client.search(collection_name=collection_name, data=vector_list, limit=limit,
                        search_params={
                            "metric_type": "COSINE"
                        }, output_fields=["origin_text"])
    result_entity = []
    result = []
    for i in res:
        for j in i:
            origin_text = j["entity"]["origin_text"]
            if origin_text in result_entity:
                continue
            result.append({"distance": j["distance"], "origin_text": origin_text})
            result_entity.append(origin_text)
    result = list(sorted(result, key=lambda x: x["distance"], reverse=True))
    return result[:topk]


if __name__ == '__main__':
    test("polyethylene")