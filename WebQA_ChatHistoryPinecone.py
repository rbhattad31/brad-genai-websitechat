from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    ConfigurableField,
    RunnableBinding,
    RunnableLambda,
    RunnablePassthrough,
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv,find_dotenv
load_dotenv(find_dotenv())

embeddings = OpenAIEmbeddings()
vectorstore = PineconeVectorStore(index_name="wbe-content", embedding=embeddings)

#vectorstore.add_texts(["i worked at kensho. my salary is 10 lakhs. i like cooking"], namespace="harrison")
#vectorstore.add_texts(["i worked at facebook"], namespace="ankush")


template = """Answer the question based only on the following context:
{context}
Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

model = ChatOpenAI()

retriever = vectorstore.as_retriever()

configurable_retriever = retriever.configurable_fields(
    search_kwargs=ConfigurableField(
        id="search_kwargs",
        name="Search Kwargs",
        description="The search kwargs to use",
    )
)

chain = (
    {"context": configurable_retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

ans = chain.invoke(
    "what is the salary?",
    config={"configurable": {"search_kwargs": {"namespace": "harrison"}}},
)
print(ans)