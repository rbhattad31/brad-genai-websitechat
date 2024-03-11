import argparse

import os
import json
import streamlit as st
from langchain.callbacks import StdOutCallbackHandler, FileCallbackHandler
from langchain.callbacks.base import BaseCallbackManager
from langchain.callbacks.manager import CallbackManager
from loguru import logger
from salesgpt.agents import SalesGPT
from langchain.chat_models import ChatOpenAI
from salesgpt.tools import get_tools, setup_knowledge_base, add_knowledge_base_products_to_cache

from salesgpt.callbackhandler import MyCustomHandler
from dotenv import load_dotenv,find_dotenv
load_dotenv(find_dotenv())
if __name__ == "__main__":

    # Initialize argparse
    parser = argparse.ArgumentParser(description='Description of your program')

    # Add arguments
    parser.add_argument('--config', type=str, help='Path to agent config file', default='')
    parser.add_argument('--verbose', type=bool, help='Verbosity', default=False)
    parser.add_argument('--max_num_turns', type=int, help='Maximum number of turns in the sales conversation', default=10)

    # Parse arguments
    args = parser.parse_args()

    # Access arguments
    config_path = args.config
    verbose = args.verbose
    max_num_turns = args.max_num_turns


    #handlers
    handler = StdOutCallbackHandler()
    customhandler=MyCustomHandler()

    logfile = "examples/output.log"

    logger.add(logfile, colorize=True, enqueue=True)
    filehandler = FileCallbackHandler(logfile)

    #llm = ChatOpenAI(temperature=0.2)
    #llm = AzureChatOpenAI(temperature=0.6, deployment_name="bradsol-openai-test", model_name="gpt-35-turbo",callbacks=[customhandler,filehandler],request_timeout=10,max_retries=3)
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo",
                     openai_api_key=os.environ.get("OPENAI_API_KEY"),
                     openai_organization=os.environ.get("OPENAI_API_ORG"))
    if not os.path.isdir('faiss_index'):
        add_knowledge_base_products_to_cache("sample_product_catalog.txt")

    if config_path=='':
        print('No agent config specified, using a standard config')
        USE_TOOLS=True
        if USE_TOOLS:
            config = dict(
                salesperson_name="LinkLynx",
                salesperson_role="Digital Navigator",
                company_name="BRADSOL",
                company_business="",
                company_values="",
                conversation_purpose="guiding users through the internal company website, providing quick access to important links, documents, and other browsable content",
                conversation_history=[],
                conversation_type="chat",
                use_tools=True,
                product_catalog="examples/sample_product_catalog.txt"
            )
            sales_agent = SalesGPT.from_llm(llm, verbose=False, **config)
        else:
            sales_agent = SalesGPT.from_llm(llm, verbose=verbose)
    else:
        with open(config_path,'r') as f:
            config = json.load(f)
        print(f'Agent config {config}')
        sales_agent = SalesGPT.from_llm(llm, verbose=verbose, **config)


    st.header('Internal FAQ Chatbot')
    # History is empty then it needs to execute

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        st.session_state.sales_agent = sales_agent
        # init sales agent
        st.session_state.sales_agent.seed_agent()
        logger.info("Init Done")

    if human := st.chat_input():
        print("\n")
        logger.info("Human "+human)
        st.session_state.chat_history.append(human)
        st.session_state.sales_agent.human_step(human)

    st.session_state.sales_agent.determine_conversation_stage()
    st.session_state.sales_agent.step()
    print("\n")