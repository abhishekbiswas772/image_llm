from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
import os

from openai import api_version


# class LLMHandler:
#     @staticmethod
#     def load_llm_model():
#         load_dotenv()
#         open_api_key = os.getenv("AZURE_OPENAI_API_KEY")
#         base_url = os.getenv("AZURE_OPENAI_ENDPOINT")
#         llm = AzureChatOpenAI(
#             azure_endpoint=base_url,
#             api_key=open_api_key,
#             api_version=os.getenv("OPENAI_API_VERSION"),
#             temperature=0.3
#         )
#         return llm


from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os


class LLMHandler:
    @staticmethod
    def load_llm_model():
        load_dotenv()
        open_api_key = os.getenv("OPEN_API_KEY")
        llm = ChatOpenAI(
            model = "gpt-4o-mini",
            api_key=open_api_key,
            temperature=0.3
        )
        return llm
    