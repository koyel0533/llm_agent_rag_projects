#llm
#tools-->Google search, web scraping, file reading, etc.
#Agents
#Memory
#Streaming
#Web Application

from dotenv import load_dotenv
load_dotenv()

import os
import streamlit as st
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

search = GoogleSerperAPIWrapper()
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))
tools = [search.run]

if "memory" not in st.session_state:
    st.session_state.memory = MemorySaver()

if "history" not in st.session_state:
    st.session_state.history = []

agent = create_agent(
    model=llm,
    tools=tools,
    checkpointer=st.session_state.memory,
    system_prompt="you are a agent can search for any question on google",
)

st.subheader("Google Search Agent")

for message in st.session_state.history:
    st.chat_message(message["role"]).markdown(message["content"])

query = st.chat_input("Enter your question: ")
if query:
    st.chat_message("user").markdown(query)
    st.session_state.history.append({"role": "user", "content": query})

    response = agent.invoke(
        {"messages": [{"role": "user", "content": query}]},
        {"configurable": {"thread_id": "1"}},
    )
    reply = response["messages"][-1].content
    st.chat_message("assistant").markdown(reply)
    st.session_state.history.append({"role": "assistant", "content": reply})
