from create_panaromic_view import create_panoramic_view, create_panoramic_view_t2i, text_to_image
from img_loader import ImgHandler
from llm_ops import LLMHandler
from qna import QNAHandler, remove_asterisks
from agents import AgentManager
from dotenv import load_dotenv
import os
from embedding_image import EmbeddingHandler
from milvus_ops import MilvisHandler
from langchain_core.messages import HumanMessage

load_dotenv()

emb = EmbeddingHandler()

def main1():
    ImgHandler.load_base_image_and_embedding()
    ImgHandler.load_segmented_image_and_embedding()
#     # ImgHandler.make_summary_and_savein_db()
    ImgHandler.create_csv_embedding()

def main2(file_path):
   return QNAHandler.make_qna(file_path)



def main3(text_input):
    result = QNAHandler.make_qna_for_text(text_input=text_input)
    print("*************** Result From Milvus ***************")
    print(result)
    print("**************************************************")
    return result


def text_to_image_pipeline(text_input):
    agent_knowledge_source = AgentManager.get_csv_agent(
        csv_knowsource_path="./data/knowledge_source.csv"
    )
    response_from_knowledge_source = agent_knowledge_source.invoke(
        text_input
    )
    response_from_knowledge_source = response_from_knowledge_source.get('output')
    prompt_temper = "what is the pnc number from this text ?" + f"'{response_from_knowledge_source}'" + "and only return the number and nothing else as output and dont assume anything. eg: **<pnc_number>** and nothing more"
    response_tamper = agent_knowledge_source.invoke(
        prompt_temper
    )
    print("\n response from llm", response_tamper)
    response_tamper =  remove_asterisks(response_tamper.get('output'))
    print("********* Response from csv **************")
    print(response_tamper)
    print("*******************************************")
    agent_sql = AgentManager.get_sql_agent(
        db_url = os.getenv("DB_URL_PNC")
    )
    prompt_sql = f"What is the image file path according to PNC number {str(response_tamper)}? and only return the image path and nothing else as output and dont assume anything. eg: **<image_path>** and nothing more"
    print(prompt_sql)
    response_from_db = agent_sql.invoke(prompt_sql)
    response_from_db = remove_asterisks(str(response_from_db.get("output")))
    # response_from_db = "/Users/abhishekbiswas/Desktop/query_with_nested_images/data/ROCKER_COVER_AND_BREATHER_page-0001.jpg"
    # response_tamper = "01110"
    result = image_search(str(response_from_db), str(response_tamper))
    new_filename = os.path.splitext(str(response_from_db).replace("./data/", ""))[0]
    new_filename = "./" + f"segmented_{new_filename}" + "/" + result 
    return new_filename
    
def image_search(file_path, pnc_number):
    MilvisHandler.connect_to_milvus()
    embedding = emb.create_embedding(
        image_path=file_path
    )
    entity = {"vector": embedding}
    result = MilvisHandler.semantic_search_version2(
        entity
    )
    print("*******************************************")
    print(result)
    print("*******************************************")
    base_image_set=set([])
    for i in result[-1]:
        print(i)
        if i['cid'] is not None and i["pid"] is not None:
            # full_base_img_path = os.path.join("./base_images", i["pid"])
            new_filename = os.path.splitext(i["pid"])[0]
            full_child_img_path = os.path.join(f"./segmented_{new_filename}", i['cid'])
            base_image_set.add(full_child_img_path)
        # base_image_set.add(full_base_img_path)
    # print(base_image_set)
    # t_img = text_to_image(f"Which images have this PNC Number {pnc_number}", "image.png")
    q_img = os.getenv("./base_images", file_path)
    panoramic_image_indexes = create_panoramic_view_t2i(retrieved_images=list(base_image_set),output_image_path="panoramic_image.jpg")
    llm = LLMHandler.load_llm_model()
    query_image = ImgHandler.load_image("panoramic_image.jpg")["image"]
    p_image = ImgHandler.load_image(q_img)["image"]

    text = (f"You will analyze two images. The first is a panoramic image of several parts, "
        "and the second is an image with arrows and labels that map these parts. "
        "Each part in the second image is a close-up of a specific section of the panoramic image. "
        f"Your task is to find the part in the panoramic image corresponding to the label {pnc_number} "
        "based on the arrows and labels provided in the second image. "
        "Return only the index of the part as an integer, without any extra information, symbols, or characters.")

    prompt = HumanMessage(
        content=[
            {"type": "text", "text": "You are an expert in image analysis and comparison. Analyze the provided images and identify the matching part based on the given PNC Number."},
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{query_image}"}} ,
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{p_image}"}}
        ]
    )


    # Invoke the LLM model
    response = llm.invoke([prompt])
    result = remove_asterisks(response.content)
    print("****************** Result from LLM *******************************")
    print(result)
    print("**************************************************")
    print("this is path of the most probable image =====>", panoramic_image_indexes[int(result)])
    print("Sleep for 60 seconds for rate or token releated issue")
    return panoramic_image_indexes[int(result)]
    



