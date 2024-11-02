from create_collection import get_client, collection_name

client = get_client()
res = client.query(collection_name=collection_name, output_fields=["count(*)"])

print(res)