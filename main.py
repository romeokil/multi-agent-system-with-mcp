import os
from typing import TypedDict,Annotated
import operator

import psycopg
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)

from langchain_groq import ChatGroq

from tools.tavily_tool import tavily_search
from tools.flight_tool import search_flights

from dotenv import load_dotenv
load_dotenv()

DATABASE_URL= os.getenv("DATABASE_URL")

llm= ChatGroq(
    model="openai/gpt-oss-20b"
)

# ye smjh le ki ek trah se nodes define kiye hmlog 
# graph me jaisa hota hai or class mtlb blueprint type ka
class TravelState(TypedDict):
    messages: Annotated[list[AnyMessage],operator.add]
    user_query: str
    flight_results: str
    hotel_results: str
    itinerary: str
    llm_calls: int


def flight_agent(state: TravelState):
    query = state["user_query"]
    flight_data = search_flights(query)
    return {
        "flight_results": flight_data,
        "messages":[
            AIMessage(content=f"Flight results fetched")
        ],
        "llm_calls": state.get("llm_calls",0) + 1
    }


def hotel_agent(state: TravelState):
    query=f"Best hotels for {state['user_query']}"
    hotel_results = tavily_search(query)

    return {
        "hotel_results": hotel_results,
        "messages": [
            AIMessage(content="Hotel Information Fetched")
        ],
        "llm_calls": state.get("llm_calls",0) + 1
    }

def itinerary_agent(state: TravelState):
    prompt = f"""
    Create a Travel Itinerary.
    User Query:
    {state["user_query"]}

    Flight Results:
    {state["flight_results"]}

    Hotel Results:
    {state["hotel_results"]}

    """

    response = llm.invoke([
        SystemMessage(
            content="You are an expert travel Planner"
        ),
        HumanMessage(content=prompt)
    ])

    return {
        "itinerary": response.content,
        "messages": [response],
        "llm_calls": state.get("llm_calls",0) + 1
    } 


def final_agent(state: TravelState):
    final_prompt = f"""
    Generate a final travel response.

    Flights:
    {state["flight_results"]}

    Hotels:
    {state["hotel_results"]}

    Itinerary:
    {state["itinerary"]}

    """

    response = llm.invoke([
        HumanMessage(content=final_prompt)
    ])

    return {
        "messages": [response],
        "llm_calls": state.get("llm_calls",0) + 1
    }

# ye hmlog graph ya workflow create kr rhe hai
graph = StateGraph(TravelState)

# ye hmlog us graph ya workflow me agents/function define kr rhe h
graph.add_node("flight_agent", flight_agent)
graph.add_node("hotel_agent", hotel_agent)
graph.add_node("itinerary_agent", itinerary_agent)
graph.add_node("final_agent", final_agent)

# workflow ya graph basically esme trigger kr rhe hai# or bol rhe hai es funtion ya agent se start and end kro
graph.add_edge(START, "flight_agent")
graph.add_edge("flight_agent","hotel_agent")
graph.add_edge("hotel_agent","itinerary_agent")
graph.add_edge("itinerary_agent","final_agent")
graph.add_edge("final_agent", END)

_conn = psycopg.connect(DATABASE_URL,autocommit=True)
checkpointer = PostgresSaver(_conn)
checkpointer.setup()

app=graph.compile(checkpointer=checkpointer)

if __name__ == "__main__":
    
    config={
        "configurable":{
            "thread_id": "user_rahul"
        }
    }

    user_input = input("Enter Your Travel Plans: ")

    results = app.invoke(
        {
            "message":[
                HumanMessage(content=user_input)
            ],
            "user_query": user_input,
            "flight_results": "",
            "hotel_results": "",
            "itinerary": "",
            "llm_calls": 0
        },
        config=config
    )


print("\n FINAL RESPONSE: \n")

for msg in results["messages"]:
    print(msg.content)
