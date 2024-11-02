from pymilvus import MilvusClient, connections
from pymilvus import FieldSchema, DataType, CollectionSchema
from tuning.config import milvus_ip
collection_name = "chemical_vector_collection"
# database_name = "chemical_vector_database"


def get_client():
    return MilvusClient(uri=f"tcp://{milvus_ip}")


if __name__ == '__main__':

    fields = []
    fields.append(FieldSchema(name="id", dtype=DataType.INT64, is_primary=True))
    fields.append(FieldSchema(name="origin_text", dtype=DataType.VARCHAR, max_length=1024))
    fields.append(FieldSchema(name="cmpdname", dtype=DataType.VARCHAR, max_length=1024))
    fields.append(FieldSchema(name="average_embedding", dtype=DataType.FLOAT_VECTOR, dim=4096))
    schema = CollectionSchema(fields=fields, auto_id=True, enable_dynamic_field=True)
    client = get_client()
    index_params = client.prepare_index_params()

    index_params.add_index(
        field_name="average_embedding",
        # index_type="GPU_IVF_FLAT",
        index_type="IVF_FLAT",
        metric_type="COSINE",
        params={
            "nlist": 1024
        }
    )

    client.create_collection(
        collection_name=collection_name,
        dimension=4096,
        schema=schema,
        index_params=index_params
    )


