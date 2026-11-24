from dotenv import load_dotenv,find_dotenv
load_dotenv(find_dotenv())
from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st
llm=ChatGoogleGenerativeAI(model="gemini-2.5-flash")

st.title("Google Gemini 2.5 Chatbot")
st.markdown("This is a simple chatbot interface using Google Gemini 2.5 model.")

if "messages" not in st.session_state:
    st.session_state.messages = []  
for message in st.session_state.messages:
    if message["role"] == "user":
        st.chat_message("user").markdown(message["content"])
    else:
        st.chat_message("ai").markdown(message["content"])


query=st.chat_input("Enter your query (or type 'exit' to quit): ")
if query:
    st.session_state.messages.append({"role": "user", "content": query})
    st.chat_message("user").markdown(query)
    res=llm.invoke(query)
    st.chat_message("ai").markdown(res.content)
    st.session_state.messages.append({"role": "ai", "content": res.content})
