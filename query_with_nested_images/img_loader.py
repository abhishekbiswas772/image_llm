import os
from langchain_milvus import Milvus
from embedding_image import EmbeddingHandler
import base64
from milvus_ops import MilvisHandler, P_DB_COLLECTION_NAME, C_DB_COLLECTION_NAME, E_DB_COLLECTION_NAME, D_DB_COLLECTION_NAME
# from dotenv import load_dotenv
# from segment_image_sam2 import fetch_segments
from langchain_community.document_loaders.csv_loader import CSVLoader
from llm_ops import LLMHandler
from langchain_core.messages import HumanMessage
import time
from prepare_data import create_sql_database
from dotenv import load_dotenv

emb = EmbeddingHandler()


class ImgHandler:
    @staticmethod
    def create_csv_embedding():
        try:
            load_dotenv()
            db_url = os.getenv("DB_URL_PNC")
            create_sql_database(image_dir="./data", csv_file="./data/knowledge_source.csv", db_url=db_url)
        except Exception as e:
            print(e)
            print("Unable to store csv data to database")


    @staticmethod
    def load_base_image_and_embedding():
        MilvisHandler.connect_to_milvus()
        MilvisHandler.drop_collection(P_DB_COLLECTION_NAME)
        base_img_path = "./base_images"
        for filename in os.listdir(base_img_path):
            file_path = os.path.join(base_img_path, filename)
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
                try:
                    features = emb.create_embedding(file_path)
                    entity = {"pid": filename, "pvector": features}
                    shape_len = entity['pvector'].shape
                    status = MilvisHandler.insert_into_db(entity=entity, dims=shape_len[0], collection_name=P_DB_COLLECTION_NAME)
                    print(status)
                except Exception as e:
                    print(e)


    @staticmethod
    def make_summary_and_savein_db():
        MilvisHandler.connect_to_milvus()
        MilvisHandler.drop_collection(E_DB_COLLECTION_NAME)
        base_img_path = "./base_images"
        llm = LLMHandler.load_llm_model()
        for filename in os.listdir(base_img_path):
            file_path = os.path.join(base_img_path, filename)
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
                base_img = ImgHandler.load_image(file_path)["image"] 
                prompt  = HumanMessage(
                    content=[
                        {
                            "type": "text",
                            "text": "Summarize the following detailed description of a complex image that contains multiple sub-images and part numbers. Ensure you don't miss any part, and create a mapping between each part number and its corresponding component."
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base_img}"}
                        }
                    ]
                )
                response = llm.invoke([prompt])
                print("***** Response of LLM about Summary of base image *********")
                print(response)
                print("************************************************************")
                summary_vector = MilvisHandler.generate_encoding(response.content)[0]
                entity = {"pid": filename, "summary": response.content, "sumvector": summary_vector}
                status = MilvisHandler.status = MilvisHandler.insert_into_db(entity=entity, dims=768, collection_name=E_DB_COLLECTION_NAME)
                print(status)
                print("\n")
                print("Sleeping for 60 seconds for rate releated issue and token releated issue")
                time.sleep(60.0)


    
            

    @staticmethod
    def load_segmented_image_and_embedding():
        MilvisHandler.connect_to_milvus()
        MilvisHandler.drop_collection(C_DB_COLLECTION_NAME)
        base_img_path = "./base_images"
        for filename in os.listdir(base_img_path):
            file_path = os.path.join(base_img_path, filename)
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
                try:
                    # fetch_segments(file_path)
                    segment_path = f"./segmented_{filename}"
                    segment_path = segment_path.rsplit('.', 1)[0]
                    print(f"segment_path: {segment_path}")
                    for fname in os.listdir(segment_path):
                        fpath = os.path.join(segment_path, fname)
                        if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
                            features = emb.create_embedding(fpath)
                            entity = {"pid": filename, "cid": fname, "cvector": features}
                            shape_len = entity['cvector'].shape
                            status = MilvisHandler.insert_into_db(entity=entity, dims=shape_len[0], collection_name=C_DB_COLLECTION_NAME)
                            print(status)

                    # # Remove the directory after processing all images
                    # shutil.rmtree(segment_path)
                    # print(f"Deleted folder: {segment_path}")

                except Exception as e:
                    print(e)

    @staticmethod
    def load_query_image_and_embedding(file_path):
        MilvisHandler.connect_to_milvus()
        features = emb.create_embedding(file_path)
        entity = {"vector": features}
        search_status = MilvisHandler.semantic_search(entity)
        # search_status = search_status[-1][0]['pid']
        return search_status
    

    @staticmethod
    def load_image(image_path) -> dict:
        """Load image from file and encode it as base64."""
        def encode_image(image_path):
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        image_base64 = encode_image(image_path)
        return {"image": image_base64}


        