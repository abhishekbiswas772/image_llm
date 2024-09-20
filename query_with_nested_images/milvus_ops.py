from pymilvus import connections, utility, CollectionSchema, DataType, FieldSchema, Collection
import numpy as np
import os
from sentence_transformers import SentenceTransformer

from sqlalchemy.orm.collections import collection

DB_USER = "minioadmin"
DB_PASSWORD = "minioadmin"
DB_HOST = "localhost"
DB_PORT = "19530"
P_DB_COLLECTION_NAME = "partEmbeddingEngine"
C_DB_COLLECTION_NAME = "segmentsEmbeddingEngine"
E_DB_COLLECTION_NAME = "textImageSummary"
D_DB_COLLECTION_NAME = "csvEmbedding"


class MilvisHandler:
    @staticmethod
    def connect_to_milvus():
        try:
            connections.connect("default", user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)
            print("Connected to Milvus.")
            return 1
        except Exception as e:
            print(f"Failed to connect to Milvus: {e}")
            raise

    @staticmethod
    def create_collection(name, fields, description, consistency_level="Strong"):
        try:
            schema = CollectionSchema(fields, description)
            collection = Collection(name, schema, consistency_level=consistency_level)
            print(f"Collection '{name}' created.")
            return collection
        except Exception as e:
            print(f"Failed to create collection: {e}")
            return None

    @staticmethod
    def insert_data(collection, entity):
        try:
            if collection.name == P_DB_COLLECTION_NAME:
                data = [
                    [entity['pid']],
                    [entity['pvector']]
                ]
            elif collection.name == C_DB_COLLECTION_NAME:
                data = [
                    [entity['pid']],
                    [entity['cid']],
                    [entity['cvector']]
                ]
            elif collection.name == E_DB_COLLECTION_NAME:
                data = [
                    [entity['pid']],
                    [entity['summary']],
                    [entity['sumvector']]
                ]
            collection.insert(data)
            collection.flush()
            print(f"Inserted data into '{collection.name}'. Number of entities: {collection.num_entities}")

            return {"collection_name": collection.name, "collection_num_entities": collection.num_entities}

        except Exception as e:
            print(f"Failed to insert data: {e}")
            return None

    @staticmethod
    def create_index(collection, field_name, index_type="IVF_FLAT", metric_type="COSINE", params={"nlist": 128}):
        try:
            index = {"index_type": index_type, "metric_type": metric_type, "params": params}
            collection.create_index(field_name, index)
            print(f"Index '{index_type}' created for field '{field_name}'.")
        except Exception as e:
            print(f"Failed to create index: {e}")

    @staticmethod
    def insert_into_db(entity, dims, collection_name):
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=90, is_primary=True, auto_id=True),
            FieldSchema(name="pid", dtype=DataType.VARCHAR, max_length=90)
        ]

        # Add the appropriate vector field based on the collection name
        if collection_name == P_DB_COLLECTION_NAME:
            fields.append(FieldSchema(name="pvector", dtype=DataType.FLOAT_VECTOR, dim=dims))
            vector_field = "pvector"
        elif collection_name == C_DB_COLLECTION_NAME:
            fields.extend([
                FieldSchema(name="cid", dtype=DataType.VARCHAR, max_length=90),
                FieldSchema(name="cvector", dtype=DataType.FLOAT_VECTOR, dim=dims)
            ])
            vector_field = "cvector"
        elif collection_name == E_DB_COLLECTION_NAME:
            fields.extend([
                FieldSchema(name="summary", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="sumvector", dtype=DataType.FLOAT_VECTOR, dim=768)
            ])
            vector_field = "sumvector"
        else:
            print(f"Invalid collection name: {collection_name}. Aborting further operations.")
            return [0, "Invalid collection name"]

        # Create the collection
        collection = MilvisHandler.create_collection(collection_name, fields, "Collection for Dev Milvus")

        if collection:
            result = MilvisHandler.insert_data(collection, entity)
            print(">>> result == ", result)
            collection.flush()
            MilvisHandler.create_index(collection, vector_field)
            return [1, result]
        else:
            print(f"Collection creation failed for {collection_name}. Aborting further operations.")
            return [0, "Collection creation failed"]

    @staticmethod
    def search_and_query(collection, query_embeddings,column_name):
        try:
            # Perform the search
            res = collection.search(
                data=query_embeddings,
                anns_field=column_name,
                param={"metric_type": "COSINE", "params": {}},
                limit=5,
                expr=None,
                output_fields=["pid"]
            )
            print(res)
            filtered_results = []
            for hits in res:
                for hit in hits:
                    entity_pid = hit.entity.get('pid')
                    similarity_score = hit.distance
                    filtered_results.append({
                        "id": hit.id,
                        "pid": entity_pid,
                        "similarity": similarity_score
                    })
            if filtered_results:
                return filtered_results
            else:
                print("No results matching beyond 80%")
                return ["No results matching beyond 80%"]
        except Exception as e:
            print(f"Search failed: {e}")
            return ["Search failed due to an error"]
        


    @staticmethod
    def search_and_query_version_text(collection, query_embeddings,column_name):
        try:
            collection_schema = collection.schema
            all_fields = [
                field.name for field in collection_schema.fields
                if field.dtype != DataType.FLOAT_VECTOR  # Exclude vector fields
            ]
            res = collection.search(
                data=query_embeddings,
                anns_field=column_name,
                param={"metric_type": "COSINE", "params": {}},
                limit=5,
                expr=None,
                output_fields=all_fields
            )
            # print(res)
            filtered_results = []
            for hits in res:
                for hit in hits:
                    entity_pid = hit.entity.get('pid')
                    similarity_score = hit.distance
                    entity_cid = hit.entity.get('cid')
                    filtered_results.append({
                        "id": hit.id,
                        "pid": entity_pid,
                        "cid": entity_cid, 
                        "similarity": similarity_score
                    })
            if filtered_results:
                return filtered_results
            else:
                print("No results matching beyond 80%")
                return ["No results matching beyond 80%"]
        except Exception as e:
            print(f"Search failed: {e}")
            return ["Search failed due to an error"]
        

    

    @staticmethod
    def drop_collection(collection_name):
        try:
            connections.connect("default", user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)
            print("Connected to Milvus.")
            utility.drop_collection(collection_name)
            print(f"Dropped collection '{collection_name}'.")
            return 1
        except Exception as e:
            print(f"Failed to drop collection: {e}")

    @staticmethod
    def semantic_search(entity, isText = False):
        try:
            results = []
            if isText == True:
                text_based_collection_list = [
                    {
                        "collection_name": E_DB_COLLECTION_NAME,
                        "column_name" : "sumvector"
                    }
                ]
                for item in text_based_collection_list:
                    collection = Collection(name=item["collection_name"])
                    collection.load()
                    print(f"Collection '{collection.name}' loaded successfully.")
                    query_embeddings = np.array(entity["vector"], dtype=np.float32).reshape(1, -1)
                    result = MilvisHandler.search_and_query(collection, query_embeddings, item["column_name"])
                    print("*********** Raw Result from milvus for text based search ***********")
                    print(result)
                    print("**********************************************************************")
                    results += result
            else:
                collection_list = [
                    {
                        "collection_name":C_DB_COLLECTION_NAME,
                        "column_name":"cvector"
                    }
                    ,
                    {
                        "collection_name":P_DB_COLLECTION_NAME,
                        "column_name":"pvector"
                    }
                ]

                for collection_data in collection_list:
                    collection = Collection(name=collection_data["collection_name"])
                    # MilvisHandler.create_index(collection, "vector", "IVF_FLAT", "COSINE", {"nlist": 128})
                    collection.load()
                    print(f"Collection '{collection.name}' loaded successfully.")
                    query_embeddings = np.array(entity["vector"], dtype=np.float32).reshape(1, -1)
                    result = MilvisHandler.search_and_query(collection, query_embeddings,collection_data["column_name"])
                    print(result)
                    results += result

            top_results = sorted(results, key=lambda x: x["similarity"], reverse=True)
            top_results =top_results[:5]
            print("Top 5 result ===>", top_results)
            return [1, top_results]
        except Exception as e:
            print(f"Failed to load collection or perform search: {e}")
            return [0, str(e)]
        

    @staticmethod
    def semantic_search_version2(entity):
        try:
            parent_collection_list = {
                "collection_name":P_DB_COLLECTION_NAME,
                "column_name":"pvector"
            }
            child_collection_list = {
                "collection_name":C_DB_COLLECTION_NAME,
                "column_name":"cvector"
            }
           
            collection_parent = Collection(name=parent_collection_list["collection_name"])
            collection_parent.load()
            print(f"Collection '{collection_parent.name}' loaded successfully.")
            query_embeddings = np.array(entity["vector"], dtype=np.float32).reshape(1, -1)
            result = MilvisHandler.search_and_query_version_text(collection_parent, query_embeddings,parent_collection_list["column_name"])
            top_results = sorted(result, key=lambda x: x["similarity"], reverse=True)
            top_results_parent = top_results[0]
            print("************** Parent Image ***********")
            print(top_results_parent)
            print("*************************************")
            pid_parent = str(top_results_parent['pid'])
            # print(pid_parent)
            collection_child = Collection(
                name=child_collection_list["collection_name"]
            )
            collection_child.load()
            query_expr = f"pid == '{pid_parent}'"
            result_child = collection_child.query(expr=query_expr, output_fields = ["pid", "cid"])
            print("********* All Child Images *********")
            print(result_child)
            print("*************************************")
            filtered_results_child = []
            for hits in result_child:
                entity_pid = hits['pid']
                entity_cid = hits['cid']
                filtered_results_child.append({
                    "pid": entity_pid,
                    "cid": entity_cid, 
                })
            return [1, filtered_results_child]
        except Exception as e:
            print(f"Failed to load collection or perform search: {e}")
            return [0, str(e)]
        
    @staticmethod
    def generate_encoding(question):
        model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')  # Changed model
        vector_embedding = model.encode(question)    
        return [vector_embedding]


