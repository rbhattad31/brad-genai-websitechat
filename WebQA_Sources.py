import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel
import os
import shutil
# Azure Details:
from dotenv import load_dotenv,find_dotenv
load_dotenv(find_dotenv())
# Load, chunk and index the contents of the blog.
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

# Retrieve and generate using the relevant snippets of the blog.
retriever = vectorstore.as_retriever()
template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.
Always say "thanks for asking!" at the end of the answer.

{context}

Question: {question}

Helpful Answer:"""
custom_rag_prompt = PromptTemplate.from_template(template)
#prompt = hub.pull("rlm/rag-prompt")
#print(prompt)
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo",
                     openai_api_key=os.environ.get("OPENAI_API_KEY"),
                     openai_organization=os.environ.get("OPENAI_API_ORG"))

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain_from_docs  = (
    RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
    | custom_rag_prompt
    | llm
    | StrOutputParser()
)

rag_chain_with_source = RunnableParallel(
    {"context": retriever, "question": RunnablePassthrough()}
).assign(answer=rag_chain_from_docs)

print(rag_chain_with_source.invoke("What is Task Decomposition"))

#for chunk in rag_chain_with_source.stream("What is Task Decomposition?"):
#    print(chunk, end="", flush=True)