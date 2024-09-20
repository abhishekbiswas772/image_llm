from langchain_experimental.agents.agent_toolkits import create_csv_agent
from llm_ops import LLMHandler
from langchain.agents.agent_types import AgentType
import logging
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities import SQLDatabase

log = logging.getLogger(name="agent_logging")


class AgentManager:
    @staticmethod
    def get_csv_agent(csv_knowsource_path=""):
        llm = LLMHandler.load_llm_model()
        if not csv_knowsource_path:
            log.error("CSV path is empty.")
            return None
        
        try:
            agent = create_csv_agent(
                llm,
                path=csv_knowsource_path,
                verbose=True,
                agent_type=AgentType.OPENAI_FUNCTIONS,
                allow_dangerous_code=True,
                return_direct = True
            )
            log.debug("Agent created successfully")
            return agent
        except Exception as e:
            log.error(f"Error in creating agent: {e}")
            return None

  
    @staticmethod
    def get_sql_agent(db_url):
        llm = LLMHandler.load_llm_model()
        if not db_url:
            log.error("DB URL is empty")
            return None
        try:
            db = SQLDatabase.from_uri(db_url)
            agent = create_sql_agent(
                llm,
                db=db,
                agent_type="openai-tools", 
                verbose=True,
            )
            print("Agent created successfully")
            return agent
        except Exception as e:
            print(f"Error creating SQL agent: {e}")
            return None
                


