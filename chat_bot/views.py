from django.shortcuts import render

# Create your views here.
from django.shortcuts import render
from django.http import JsonResponse

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage

from typing import TypedDict, Annotated
import os

class ChatState(TypedDict):
    message: Annotated[list[BaseMessage], add_messages]
model=ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key="AIzaSyBSUnlUr-74-ZawaQw47RUKLWpyAam_i4s")
def chat_node(state: ChatState):
    messages = state["message"]
    response = model.invoke(messages)
    return {"message": [response]}

checkpointer = MemorySaver()

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)
CONFIG={'configurable':{'thread_id':'1'}}
chatbot = graph.compile(checkpointer=checkpointer)
def chat_view(request):
    if request.method == "POST":
        user_message = request.POST.get("message", "")
        initial_state = {"message": [HumanMessage(content=user_message)]}
        result = chatbot.invoke(initial_state,config=CONFIG)
        bot_response = result["message"][-1].content
        return JsonResponse({"response": bot_response})
    return render(request, "chat.html")