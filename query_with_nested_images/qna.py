from llm_ops import LLMHandler
from img_loader import ImgHandler
from langchain_core.messages import HumanMessage
from milvus_ops import MilvisHandler, D_DB_COLLECTION_NAME
import os
import pandas as pd
import cv2
import numpy as np
from create_panaromic_view import * 
from langchain_milvus import Milvus
from langchain_huggingface import HuggingFaceEmbeddings
import re
import time

emb = HuggingFaceEmbeddings()


def remove_asterisks(text):
    if text.startswith('**') and text.endswith('**'):
        return text[2:-2]
    return text


def find_details_from_database(part_name):
    csv_path = "./car_engine_parts.csv"
    df = pd.read_csv(csv_path)
    part_name = part_name.lower()

    result = df[df['partname'].str.lower().str.contains(part_name, na=False)]

    if not result.empty:
        return result[['id', 'partname', 'description']]
    else:
        return pd.DataFrame(columns=['id', "partname", 'description'], index=None)


class QNAHandler:
    @staticmethod
    def build_prompt(text, main_image, query_image):
        message = HumanMessage(
            content=[
                {"type": "text",
                 "text": "You are an expert algorithm for comparing images. Your task is to identify and find the number associated to the query image from the text present in the main image."},
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{main_image}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{query_image}"}},
            ]
        )
        return message

    @staticmethod
    def window_slide_strategy(query_image_path, result, base_dir, threshold=0.90):
        query_image = cv2.imread(query_image_path, cv2.IMREAD_GRAYSCALE)
        if query_image is None:
            print(f"Failed to load query image: {query_image_path}")
            return None

        query_image = cv2.GaussianBlur(query_image, (5, 5), 0)
        query_image = cv2.Canny(query_image, 100, 200)
        for res in result[-1]:
            try:
                main_file_path = os.path.join(base_dir, res['pid'])
                parent_image = cv2.imread(main_file_path, cv2.IMREAD_GRAYSCALE)
                if parent_image is None:
                    print(f"Failed to load image: {main_file_path}")
                    continue

                parent_image = cv2.GaussianBlur(parent_image, (5, 5), 0)
                parent_image = cv2.Canny(parent_image, 100, 200)

                for scale in [1.0, 0.9, 0.8, 0.7]:
                    resized_query = cv2.resize(query_image, (0, 0), fx=scale, fy=scale)
                    result = cv2.matchTemplate(parent_image, resized_query, cv2.TM_CCOEFF_NORMED)
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                    print(f"Match results for {res['pid']} at scale {scale}: Min: {min_val:.4f}, Max: {max_val:.4f}")
                    if max_val >= threshold:
                        print(f"Match found in image: {res['pid']} with match value: {max_val:.4f}")
                        return res['pid']
            except Exception as e:
                print(f"Error processing {res['pid']}: {str(e)}")
        return None

    @staticmethod
    def is_drawing(image_path, edge_threshold=0.05, contrast_threshold=0.5):
        image = cv2.imread(image_path)
        if image is None:
            print("Error: Unable to load the image.")
            return False
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_image, 100, 200)
        edge_proportion = np.sum(edges > 0) / edges.size
        print(f"Edge Proportion: {edge_proportion:.4f}")
        if edge_proportion > edge_threshold:
            print("The image has prominent edges, suggesting it might be a drawing.")
            return True
        _, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        white_pixels = np.sum(binary_image == 255)
        black_pixels = np.sum(binary_image == 0)
        contrast_proportion = min(white_pixels, black_pixels) / (white_pixels + black_pixels)
        print(f"Contrast Proportion: {contrast_proportion:.4f}")
        if contrast_proportion > contrast_threshold:
            print("The image has high contrast, suggesting it might be a drawing.")
            return True

        print("The image is likely not a drawing based on both edge and contrast analysis.")
        return False

    @staticmethod
    def remove_stars(text_content):
        pattern = r'\*\*(.+?)\*\*'
        extracted_content = re.search(pattern, text_content).group(1)
        return extracted_content

    @staticmethod
    def make_qna(image_file):
        base_path = "./base_images"
        print(image_file)
        probable_img_image = ImgHandler.load_query_image_and_embedding(file_path=image_file)
        print("******** Result from milvus ************")
        print(probable_img_image)
        print("******************************************")
        base_image_set=set([])
        for i in probable_img_image[1]:
            full_base_img_path = os.path.join(base_path, i["pid"])
            base_image_set.add(full_base_img_path)
        print("This is the most probable image we got till now ===>", base_image_set)
        panoramic_image_indexes=create_panoramic_view(query_image_path=image_file,retrieved_images=list(base_image_set),output_image_path="panoramic_image.jpg")
        print("This is panoramic image view having images as index ===>", panoramic_image_indexes)
        llm = LLMHandler.load_llm_model()
        query_image = ImgHandler.load_image("panoramic_image.jpg")["image"]
        text="there are many images in marked in red box these are the base image and there is one query image passed in blue color you have to tell from which base image the query image belongs to tell the index only give in integer index do not give any special characters or any other characters eg <index of base image>"
        prompt = HumanMessage(
            content=[
                {"type": "text",
                    "text": "You are an expert algorithm for comparing images. Your task is to identify and find the base image of the query image from the text present in the main image."},
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{query_image}"}},
            ]
        )
        response = llm.invoke([prompt])
        print("*************************************************")
        print(response)
        print("**************************************************")
        result = remove_asterisks(response.content)
        print("this is path of the most probable image =====>", panoramic_image_indexes[int(result)])
        print("Sleep for 60 seconds for rate or token releated issue")
        time.sleep(60.0)
        try:
            main_image_result = ImgHandler.load_image(os.path.join("./base_images", panoramic_image_indexes[int(result)]))["image"]
            query_image_result = ImgHandler.load_image(image_file)["image"]
            prompt_get_name = QNAHandler.build_prompt(
                "Your task to get the full part number in the base image assocaited to the part of image given, The part number is the mixture of english digit and Japanese, i need the exact number **<Part number>**",
                main_image_result,
                query_image_result
            )
            response_from_llm = llm.invoke([prompt_get_name])
            result_after_removing = QNAHandler.remove_stars(response_from_llm.content)
            print(result_after_removing)
            return (panoramic_image_indexes[int(result)], result_after_removing)
        except Exception as e:
            print(e)

    @staticmethod
    def make_qna_for_text(text_input):
        MilvisHandler.connect_to_milvus()
        base_path = "./base_images"
        print("********* User Input Text ***********")
        print(text_input)
        print("**************************************")
        embedding_text_input = MilvisHandler.generate_encoding(text_input)[0]
        print(embedding_text_input.shape)
        entity = {"vector": embedding_text_input}
        search_status = MilvisHandler.semantic_search(entity, isText=True)
        # print(search_status)
        base_image_set=set([])
        for i in search_status[-1]:
            full_base_img_path = os.path.join(base_path, i["pid"])
            base_image_set.add(full_base_img_path)
        print("This is the most probable image we got till now ===>", base_image_set)
        encoded_image_path = text_to_image(text_input, "image.png")
        panoramic_image_indexes = create_panoramic_view(
            query_image_path = encoded_image_path,
            retrieved_images = list(base_image_set),
            output_image_path="panoramic_image.jpg"
        )
        print("This is panoramic image view having images as index ===>", panoramic_image_indexes)
        llm = LLMHandler.load_llm_model()
        query_image = ImgHandler.load_image("panoramic_image.jpg")["image"]
        text="there are many images in marked in red box these are the base image and there is one query text passed in blue color you have to tell from which base image the query text belongs to tell the index only give in integer index do not give any special characters or any other characters eg <index of base image>"
        prompt = HumanMessage(
            content=[
                {"type": "text",
                    "text": "You are an expert algorithm for comparing images and answer question. Your task is to identify and find the base image of the query text from the text present in the main image."},
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{query_image}"}},
            ]
        )
        response = llm.invoke([prompt])
        print("*************************************************")
        print(response)
        print("**************************************************")
        result = remove_asterisks(response.content)
        print("*************** Result from Azure *******************")
        print(result)
        print("**************************************************")
        print("this is path of the most probable image =====>", panoramic_image_indexes[int(result)])
        return panoramic_image_indexes[int(result)]
    

    @staticmethod
    def make_qna_with_csv(text_input):
        pass



        