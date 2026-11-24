from dotenv import load_dotenv,find_dotenv
load_dotenv(find_dotenv())
from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st
llm=ChatGoogleGenerativeAI(model="gemini-2.5-flash")
'''
while True:
    query = input("Enter your query (or type 'exit' to quit): ")
    if query.lower() in ['exit', 'quit']:
        print("Exiting the program.")
        break
    response = llm.invoke(query)
    print(response.content)
'''