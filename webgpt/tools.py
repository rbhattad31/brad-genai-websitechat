from langchain import FAISS, GoogleSerperAPIWrapper, SerpAPIWrapper
from langchain.agents import Tool
from langchain.chains import RetrievalQA
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import AzureOpenAI
from langchain.document_loaders import TextLoader
from loguru import logger
from salesgpt.logger import time_logger
from webgpt.import_data import login
from webgpt.import_data import get_data
import os
from Config import create_main_config_dictionary
@time_logger
def add_knowledge_base_to_store():
    """
        We assume that the product catalog is simply a text string.
        """
    # load the document and split it into chunks
    # Load, chunk and index the contents of the blog.
    config_path = 'WebChat_Config.xlsx'
    sheet = 'Sheet1'
    config_dict,status = create_main_config_dictionary(config_path,sheet)
    chrome_driver_path = config_dict['chrome_driver_path']
    output_file_directory = config_dict['output_directory']
    if not os.path.exists(output_file_directory):
        os.makedirs(output_file_directory)
    url = config_dict['url']
    username = config_dict['username']
    password = config_dict['password']
    links = str(config_dict['links']).split(',')
    login_status, driver = login(chrome_driver_path, url, username, password)
    if login_status:
        data, output_file_paths = get_data(driver, output_file_directory, links)
        if data:
            print("Successfully extracted data from Site")
        else:
            output_file_paths = []
    else:
        output_file_paths = []
    if len(output_file_paths) !=0:
        with open('dummy_file.txt', 'w') as file:
            # No need to write anything, as the file is empty
            file.write('Below is the data extracted from websites')
        loader = TextLoader('dummy_file.txt', encoding='UTF-8')
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)
        embeddings = OpenAIEmbeddings()
        master_db = FAISS.from_documents(texts, embeddings)
        for i,file_path in enumerate(output_file_paths):
            loader = TextLoader(file_path, encoding='UTF-8')
            documents = loader.load()
            text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=0)
            texts = text_splitter.split_documents(documents)
            embeddings = OpenAIEmbeddings()
            db = FAISS.from_documents(texts, embeddings)
            master_db.merge_from(db)
            os.remove(file_path)
        master_db.save_local("faiss_index")
        os.remove('dummy_file.txt')
