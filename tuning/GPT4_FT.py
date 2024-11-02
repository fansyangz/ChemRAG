import json

headers = {"Authorization": "Bearer sk-5LkhJqE3f3weksQQAcFSMIsNkghYOs1hRSQPpyIIz6T3BlbkFJJDK5WB8FnOINdqh10ZqNbX1KG9sNegumeS0oVeNDwA"}
import requests

proxies = {
    "http": "http://127.0.0.1:7890",
    "https": "http://127.0.0.1:7890",
}

def by_requests():
    url = "https://api.openai.com/v1/files"
    with open('train_answer.jsonl', "rb") as f:
        files = {
            "purpose": (None, "fine-tune"),

            "file": f
        }
        response = requests.post(url=url, headers=headers, files=files, proxies=proxies)
        print(response.json())
    with open('dev_answer.jsonl', "rb") as f:
        files = {
            "purpose": (None, "fine-tune"),

            "file": f
        }
        response = requests.post(url=url, headers=headers, files=files, proxies=proxies)
        print(response.json())
    with open('test_answer.jsonl', "rb") as f:
        files = {
            "purpose": (None, "fine-tune"),

            "file": f
        }
        response = requests.post(url=url, headers=headers, files=files, proxies=proxies)
        print(response.json())

        # {'object': 'file', 'id': 'file-awnzS8se31QKBtV9cpZHzehc', 'purpose': 'fine-tune',
        #  'filename': 'train_answer.jsonl', 'bytes': 1224866, 'created_at': 1726668390, 'status': 'processed',
        #  'status_details': None}
        # {'object': 'file', 'id': 'file-ZWz2HnXpsqZkc9lAzFwdnVSx', 'purpose': 'fine-tune',
        #  'filename': 'dev_answer.jsonl', 'bytes': 71768, 'created_at': 1726668393, 'status': 'processed',
        #  'status_details': None}
        # {'object': 'file', 'id': 'file-BoGIXrgOPqwUwBZOnrWrQz8l', 'purpose': 'fine-tune',
        #  'filename': 'test_answer.jsonl', 'bytes': 137476, 'created_at': 1726668394, 'status': 'processed',
        #  'status_details': None}

def create_ft_job():
    url = "https://api.openai.com/v1/fine_tuning/jobs"
    headers["Content-Type"] = "application/json"
    data = {
        "training_file": "file-awnzS8se31QKBtV9cpZHzehc",
        "model": "gpt-4o-mini-2024-07-18"
    }
    response = requests.post(url=url, headers=headers, data=json.dumps(data), proxies=proxies)
    print(response.json())

# by_requests()
create_ft_job()