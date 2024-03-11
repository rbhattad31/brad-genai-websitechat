from langchain import FAISS, GoogleSerperAPIWrapper, SerpAPIWrapper
from langchain.agents import Tool
from langchain.chains import RetrievalQA
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import AzureOpenAI
from langchain.document_loaders import TextLoader
from loguru import logger
from salesgpt.logger import time_logger

@time_logger
def add_knowledge_base_to_store():
    """
        We assume that the product catalog is simply a text string.
        """
    # load the document and split it into chunks
    # Load, chunk and index the contents of the blog.
    loader = TextLoader("examples/about.txt", encoding='UTF-8')
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(texts, embeddings)
    db.save_local("faiss_index")