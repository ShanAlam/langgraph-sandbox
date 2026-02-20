from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END 
from langchain_core.messages import HumanMessage 
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

class StateSchema(TypedDict): 
    messages: List[HumanMessage]

llm = ChatOpenAI(model='gpt-4o-mini')

def process(state:StateSchema) -> StateSchema: 
    response = llm.invoke(state['messages'])
    print(response.content)
    return state

graph = StateGraph(StateSchema)

graph.add_node('process_node', process)

graph.add_edge(START, 'process_node')
graph.add_edge('process_node', END)

app = graph.compile()

user_input = input('What can i do for you today?')
while user_input != 'exit': 
    app.invoke({'messages': [HumanMessage(content=user_input)]})
    user_input = input('Anything else i can help you with?')