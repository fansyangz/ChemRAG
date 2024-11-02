import json
result_list = []
with open("result/result_rag.txt", "r") as file:
    for e in file.readlines():
        if e == "\n":
            continue
        result_list.append(json.loads(e)[0])
with open("result/result_rag.txt", "w") as file:
    for result in result_list:
        file.write(result)