from input_embedding import from_word_to_embeddings, es, index_name

def get_query_json(query_vector):
    return {
        "query": {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": "cosineSimilarity(params.queryVector, 'average_embedding') + 1.0",
                    "params": {
                        "queryVector": query_vector
                    }
                }
            }
        }
    }


text = "7-(2-(4-(O-METHOXYPHENYL)-1-PIPERAZINYL)ETHYL)-5H-1,3-DIOXOLO(4,5-F)INDOLE"
# text = "Solypertine"

response = es.search(index=index_name, body=get_query_json(from_word_to_embeddings(text)["average_embedding"]))
print(response)
