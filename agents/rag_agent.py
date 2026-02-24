from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from typing import TypedDict, Sequence, Annotated, List, Dict 
from dotenv import load_dotenv 
from langgraph.graph.message import add_messages 
import os

from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# ================================================================
# Creating Vector DB 
# ================================================================

load_dotenv()

# load the embedder 
embedder = OpenAIEmbeddings(model="text-embedding-3-small")

# check if pdf exists 
pdf_path = "./agents/Stock_Market_Performance_2024.pdf"
if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"PDF not found: {pdf_path}")

# Initialise loader
pdf_loader = PyPDFLoader(pdf_path)

# load in the pdf
pages = pdf_loader.load()

# initialise splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

# split pdf into chunks 
chunks = text_splitter.split_documents(pages)

# set the location you want the vector db 
persist_directory = r"/Users/shan/Projects/langgraph-sandbox/agents"

# set the name of the vectorDB
collection_name = "stock_market"

# If our collection does not exist in the directory, we create using the os command
if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)

# initialise the actual vector DB
try:
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedder,
        persist_directory=persist_directory,
        collection_name=collection_name
    )
    print(f"Created ChromaDB vector store!")
    
except Exception as e:
    print(f"Error setting up ChromaDB: {str(e)}")
    raise


# create retriever interface (defines the way you will retrieve from the DB)
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5} # K is the amount of chunks to return
)


# ================================================================
# Back to normal flow 
# ================================================================


# load the llm 
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Create a tool which will allow retrival from the VectorDB
@tool 
def retrival_tool(query:str) -> str: 
    """This tool searches and returns the information from the Stock Market Performance 2024 document."""

    docs = retriever.invoke(query)

    if not docs:
        return "No relevant context found in document."
    
    return "\n\n".join(doc.page_content for doc in docs)

my_tools = [retrival_tool]

llm = llm.bind_tools(my_tools)

# Create state schema
class StateSchema(TypedDict): 
    messages: Annotated[Sequence[BaseMessage], add_messages]

# Create agent function
def agent_func(state: StateSchema) -> StateSchema:

    system_prompt = SystemMessage(
        content=(
            "You are an intelligent AI assistant who answers questions about stock "
            "market performance in 2024 based on the loaded PDF document.\n"
            "Use the retriever tool to answer questions about stock market data. "
            "You may make multiple tool calls if needed.\n"
            "If you need to look up information before a follow-up question, you are "
            "allowed to do that.\n"
            "Always cite the specific parts of the document you used."
        )
    )

    all_messages = [system_prompt] + list(state["messages"])

    result = llm.invoke(all_messages)

    return {"messages": [result]}

# Create edge function 
def edge_func(state:StateSchema) -> StateSchema:
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "continue"
    else:
        return "end"

# Create the graph 
graph = StateGraph(StateSchema)

graph.add_node("agent_node", agent_func)

tool_node = ToolNode(tools=my_tools)
graph.add_node("tool_node", tool_node)

graph.add_edge(START, "agent_node")
graph.add_conditional_edges(
    "agent_node", 
    edge_func, 
    {
        "continue": "tool_node",
        "end": END
    }
)

graph.add_edge("tool_node", "agent_node")

app = graph.compile()

def running_agent():
    print("\n=== RAG AGENT===")
    
    while True:
        user_input = input("\nWhat is your question: ")
        if user_input.lower() in ['exit', 'quit']:
            break
            
        messages = [HumanMessage(content=user_input)] # converts back to a HumanMessage type

        result = app.invoke({"messages": messages})
        
        print("\n=== ANSWER ===")
        print(result['messages'][-1].content)


if __name__ == "__main__":
    running_agent()
