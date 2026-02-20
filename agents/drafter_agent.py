from typing import TypedDict, Sequence, Annotated
from langgraph.graph import START, END, StateGraph
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool 
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages 
from dotenv import load_dotenv 

load_dotenv()

class SessionState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

final_message = ""

@tool
def update_func(draft: str) -> str: 
    """This function updates the final_message"""
    global final_message
    final_message = draft 
    return f"Your document has been updated with the content: \n{final_message}"

@tool 
def save_func(name: str) -> str: 
    """This function saves the final_message inside  .txt file"""
    global final_message 

    if not(name.endswith(".txt")):
        name = f"{name}.txt"

    try:

        with open(name, 'w') as file: 
            file.write(final_message)
        
        print(f"saved as {name}!")
        return f"saved as {name}!"
    
    except Exception as e: 
        return f"Error saving document: {e}"

tools = [update_func, save_func]

llm = ChatOpenAI(model="gpt-4o-mini").bind_tools(tools)

def agent_func(state: SessionState) -> SessionState: 

    system_prompt = SystemMessage(content = f"""
    You are a drafter, a helpful writing assisstant. You are going to helpp the user update the document. 
    
    - If the user wants to modify the content of the document, use the 'update' tool. 
    - If the user wants to save the content of the document, save it as a .txt file using the 'save' tool. 
    """) 

    new_message = input('How would you like to update the document?')
    new_message = HumanMessage(content=new_message)

    new_responce = llm.invoke([system_prompt] + list(state['messages']) + [new_message])

    print(f"\n🤖 AI: {new_responce.content}")

    if hasattr(new_responce, "tool_calls") and new_responce.tool_calls: 
        print(f"Tools used: {[i['name'] for i in new_responce.tool_calls]}")

    return {'messages': list(state['messages']) + [new_message, new_responce]}

tool_node = ToolNode(tools=tools)

def edge_func(state: SessionState) -> str: 
    if not state['messages']:
        return "continue"
    
    for message in reversed(state['messages']):
        if (isinstance(message, ToolMessage) and
            "saved" in message.content.lower()): 
            return "end"
        
    return "continue"

graph = StateGraph(SessionState)

graph.add_node("agent_node", agent_func)
graph.add_node("tool_node", tool_node) 

graph.add_edge(START, "agent_node")
graph.add_edge("agent_node", "tool_node")
graph.add_conditional_edges(
    "tool_node",
    edge_func,
    {
        "continue": "agent_node",
        "end": END
    }
)

app = graph.compile()

def run_document_agent():
    print("\n ===== DRAFTER =====")
    
    state = {"messages": []}
    
    for step in app.stream(state, stream_mode="values"):
        if "messages" in step:
            if not step["messages"]:
                pass
            for message in step["messages"][-3:]:
                if isinstance(message, ToolMessage):
                    print(f"\n🛠️ TOOL RESULT: {message.content}")
    
    print("\n ===== DRAFTER FINISHED =====")


if __name__ == "__main__":
    run_document_agent()

    










