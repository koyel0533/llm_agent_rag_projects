from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import InMemoryVectorStore
from langchain.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
import os


#load the documents

loader=PyPDFLoader("../AI_Vehicle_Management_System_Interview_Guide.pdf")
docs=loader.load()
#print(len(docs))

#Splitter into document

splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
split_data=splitter.split_documents(documents=docs)

#Embeddings and data store
embeddings=GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")
vector_store=InMemoryVectorStore.from_documents(
    documents=split_data,
    embedding=embeddings
  
)
#doc=vector_store.similarity_search("patient name")


#print(doc[0].page_content)
#create a agent-->tool,llm,prompt
llm=ChatGroq(model="openai/gpt-oss-20b", groq_api_key=os.getenv("GROQ_API_KEY"))

@tool
def retrieve_context(query:str):
    '''
    Retrieve document relevant to a query from the knowledge base.

    '''
    context=""
    docs=vector_store.similarity_search(query=query,k=4)
    
        
    for doc in docs:
            context+=doc.page_content+"\n"
    return context


System_prompt="""
           You are a helpful assistant and provide answered based on the context for user questions using retreived context.
         Always use the `retrieve_context` tool for questions requiring external knowledge


          """
memory=InMemorySaver()
agent=create_agent(
      model=llm,
      tools=[retrieve_context],
      system_prompt=System_prompt,
      checkpointer=memory
)

while True:
      query=input("user: ")
      if query.lower()=='quit':
            break
      response=agent.invoke({"messages":[{"role":"user","content":query}]},
                            {"configurable":{"thread_id":1}}
                            )
      print("AI",response["messages"][-1].content)



