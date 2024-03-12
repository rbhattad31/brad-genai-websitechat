
from langchain import hub
from langchain_community.vectorstores.faiss import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import AgentExecutor, create_openai_tools_agent
import streamlit as st
import os

from dotenv import load_dotenv,find_dotenv
load_dotenv(find_dotenv())
if __name__ == "__main__":
    st.set_page_config(page_title="Website Chat", page_icon=":books:")
    st.session_state.chat_history = []

    if not os.path.isdir('faiss_index'):
        print("Creating Index")
        #add_knowledge_base_to_store()

    # Retrieve and generate using the relevant snippets of the blog.
    if os.path.isdir('faiss_index'):
        st.write("Website Content Loaded", unsafe_allow_html=True)
        embeddings = OpenAIEmbeddings()
        db = FAISS.load_local("faiss_index", embeddings)
        retriever = db.as_retriever()
        tool = create_retriever_tool(
            retriever,
            "search_site",
            "Searches and returns excerpts from Site Content.",
        )
        tools = [tool]

        prompt = hub.pull("hwchase17/openai-tools-agent")
        # print(prompt.messages)

        llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo",
                         openai_api_key=os.environ.get("OPENAI_API_KEY"),
                         openai_organization=os.environ.get("OPENAI_API_ORG"))

        agent = create_openai_tools_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools)
        #print(result["output"])
        st.header('Internal Website Chatbot')
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
            st.session_state.sales_agent = agent
            st.session_state.chat_history.append("Hello, How are you?")
        if human := st.chat_input():
            print("\n"+human)
            st.session_state.chat_history.append(human)
            result = agent_executor.invoke({"input": human})
            st.session_state.chat_history.append(result["output"])

        for i, msg in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.chat_message('user').write(msg)
            else:
                st.chat_message('assistant').write(msg)
    else:
        print("Index NOt Found")
        st.write("Kindly Load Website Content", unsafe_allow_html=True)