from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from typing import TypedDict, Sequence, Annotated, List, Dict 
from dotenv import dot_env 
from langgraph.graph.message import add_messages 

from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# load the llm 

# load the embedding model 

# load in the pdf

# double check the pdf is there by verifying the number of pages 

# define the chunker i.e. chunk size and overlap 

# apply chunker to the pdf 

# define the location of the vector database 

# create the vector database 

# 
tools = ["retriever_tool]

tools_dict = {our_tool.name: our_tool for our_tool in tools} 