from create_collection import get_client, collection_name


if __name__ == '__main__':
    # res = client.delete(
    #     collection_name=collection_name,
    #     filter="id not in [1]"
    # )
    # print(res)
    client = get_client()
    res = client.drop_collection(
        collection_name=collection_name
    )
    print(res)

