from llama3_evaluation import file_path_test, file_path_train, file_path_dev
def get_jsonl():
    import jsonlines
    from llama3_tune_source import data_process_ie
    with jsonlines.open("test_answer.jsonl", "w") as writer:
        for item in data_process_ie(file_path_test, has_answer=True):
            item = {"messages": item}
            writer.write(item)
    with jsonlines.open("train_answer.jsonl", "w") as writer:
        for item in data_process_ie(file_path_train, has_answer=True):
            item = {"messages": item}
            writer.write(item)
    with jsonlines.open("dev_answer.jsonl", "w") as writer:
        for item in data_process_ie(file_path_dev, has_answer=True):
            item = {"messages": item}
            writer.write(item)


def do_request():
    from zhipuai import ZhipuAI
    from llama3_tune_source import data_process_ie
    client = ZhipuAI(api_key="02f5fcf8e84196c9ad7477d5455057b7.TZlRdpmXxXVds3r2")
    test_data = data_process_ie(file_path_test, has_answer=False)
    result = []
    for item in test_data:
        response = client.chat.completions.create(model="glm-4-flash:1182629604::tvz2leer", messages=item, top_p=0.99, max_tokens=1024)
        result.append(response.choices[0].message.content)
    with open("glm4_result.txt", "w") as file:
        for r in result:
            file.write(r + "\n")
    print("done!")


def do_inferrence(request):
    # pip install zhipuai 请先在终端进行安装

    from zhipuai import ZhipuAI

    client = ZhipuAI(api_key="02f5fcf8e84196c9ad7477d5455057b7.TZlRdpmXxXVds3r2")  # 填写您自己的APIKey
    response = client.chat.completions.create(
        model="charglm-3",
        messages=[
            {
                "role": "user",
                "content": request
            }
        ],
        temperature=0.9,
        top_p=0.7,
        meta={
            "bot_info": "专业的文档分析助手，专门生成公司的财政分析文档。",
            "bot_name": "苏东坡",
            "user_info": "用户",
            "user_name": "用户"
        },
        stream=False
    )
    return response.choices[0].message.content
    # for trunk in response:
    #     print(trunk)


if __name__ == "__main__":
    result = do_inferrence("请生成一个数据分析报告，关于一个公司第一年收入2000万，第二年收入3000万，第三年收入1000万，"
                           "第一章是主题，第二章是数据分析内容，第三章是结论。")
    print("生成结果:", result)