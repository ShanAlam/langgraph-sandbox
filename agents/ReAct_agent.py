from typing import TypedDict, Sequence, Annotated
from langgraph.graph import START, END, StateGraph
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool 
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages 
from dotenv import load_dotenv 

load_dotenv()

class StateSchema(TypedDict): 
    messages: Annotated[Sequence[BaseMessage], add_messages]

# ================================================================
# Tools 
# ================================================================

@tool
def add_func(a:int, b:int):
    """This function will add two numbers"""

    return a + b

@tool
def subtract_func(a:int, b:int):
    """This function will subtract two numbers"""

    return a - b

@tool
def multiply_func(a:int, b:int):
    """This function will multiply two numbers"""

    return a * b

my_tools = [add_func, subtract_func, multiply_func]

llm = ChatOpenAI(model='gpt-4o-mini').bind_tools(my_tools)

# ================================================================
# Node Functions 
# ================================================================

def agent_func(state:StateSchema) -> StateSchema: 
    system_prompt = SystemMessage(
        content = "You are an AI assistant who is very good at maths"
    )

    result = llm.invoke([system_prompt] + state['messages'])

    return {'messages': [result]}

def edge_func(state:StateSchema): 
    if not state['messages'][-1].tool_calls:
        return 'end'
    else: 
        return 'continue'


tool_node = ToolNode(tools=my_tools)


# ================================================================
# Creating Graph 
# ================================================================

graph = StateGraph(StateSchema)

graph.add_node('agent_node', agent_func)

graph.add_node('tool_node', tool_node)

graph.add_edge(START, 'agent_node')

graph.add_conditional_edges(
    'agent_node',
    edge_func, 
    {
        'end':END,
        'continue':'tool_node'
    }
)

graph.add_edge('tool_node', 'agent_node')

app = graph.compile()


def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

inputs = {"messages": [("user", "Can you add 4 and 5 and the multiply by 3")]}
print_stream(app.stream(inputs, stream_mode="values"))


# result = app.invoke({'messages': HumanMessage(content='Can you add 4 and 5 and the multiply by 3')})
# print(result['messages'][-1].content)