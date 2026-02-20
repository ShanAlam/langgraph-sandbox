from typing import TypedDict, List, Dict, Union
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model='gpt-4o-mini')

class StateSchema(TypedDict): 
    message: List[Union[HumanMessage, AIMessage]]

def process(state:StateSchema) -> StateSchema: 
    output = llm.invoke(state['message'])
    print(f'\n{output.content}')
    state['message'].append(AIMessage(content=output.content))
    return state

graph = StateGraph(StateSchema)
graph.add_node('process_node', process)
graph.add_edge(START, 'process_node')
graph.add_edge('process_node', END)
app = graph.compile()

memory = []

user_input = input('What can I help you with today? \n')
while user_input.lower() != 'exit': 
    memory.append(HumanMessage(content=user_input))
    result = app.invoke({'message': memory})
    memory = result['message'] 
    user_input = input('Enter: \n')

with open('logging.txt', 'w') as file: 
    file.write('Your conversation log:\n')

    for message in memory: 
        if isinstance(message, HumanMessage): 
            file.write(f'You: {message}\n')
        elif isinstance(message, AIMessage): 
            file.write(f'AI: {message}\n')
    
    file.write('End of conversation.')

print(f'\nconversation saved to logging.txt')