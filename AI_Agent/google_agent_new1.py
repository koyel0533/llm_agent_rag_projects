from dotenv import load_dotenv
load_dotenv()
from langchain_community.utilities import GoogleSerperAPIWrapper
import os
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
search=GoogleSerperAPIWrapper()
llm=ChatGroq(model="openai/gpt-oss-20b", groq_api_key=os.getenv("GROQ_API_KEY"))

agent=create_agent(model=llm, tools=[search.run], system_prompt="you are a agent can search for any question on google",checkpointer=MemorySaver())

while True:
    query=input("enter your question: ")
    if query.lower()=="exit":
        print("Exiting the agent.")
        break
    response=agent.invoke({"messages": [{"role": "user", "content": query}]},{"configurable": {"thread_id": "1"}})
    print("Agent response:", response["messages"][-1].content)