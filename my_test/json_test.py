import pickle
aaa = {
    "a": "1",
    "b": "2"
}

with open("json_test.pkl", "wb") as file:
    pickle.dump(aaa, file)

with open("json_test.pkl", "rb") as file:
    bbb = pickle.load(file)

print(bbb)