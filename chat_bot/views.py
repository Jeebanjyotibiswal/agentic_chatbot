from django.shortcuts import render
from django.http import JsonResponse, StreamingHttpResponse
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage
from typing import TypedDict, Annotated
import os
import json
import time

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

class ChatState(TypedDict):
    message: Annotated[list[BaseMessage], add_messages]

# Initialize LLM with API key from environment

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    google_api_key=os.getenv("google_api_key"),
    temperature=0.7
)

def chat_node(state: ChatState):
    messages = state["message"]
    response = model.invoke(messages)
    return {"message": [response]}

# Create graph
checkpointer = MemorySaver()
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

CONFIG = {'configurable': {'thread_id': '1'}}
chatbot = graph.compile(checkpointer=checkpointer)

def home_view(request):
    """Render the chat interface"""
    return render(request, 'chat.html')

def chat_stream_view(request):
    """Handle streaming chat responses"""
    if request.method == 'GET':
        user_message = request.GET.get("message", "").strip()
    else:
        user_message = request.POST.get("message", "").strip()
    
    if not user_message:
        return JsonResponse({"error": "No message provided"}, status=400)

    def event_stream():
        try:
            # Initial state with user message
            initial_state = {
                "message": [HumanMessage(content=user_message)]
            }

            # Stream responses from LangGraph
            stream = chatbot.stream(
                initial_state,
                config=CONFIG,
                stream_mode="messages"
            )
            
            full_response = ""
            
            for message_chunk, metadata in stream:
                if hasattr(message_chunk, 'content') and message_chunk.content:
                    chunk_text = message_chunk.content
                    full_response += chunk_text
                    
                    # Send each chunk as SSE
                    data = {
                        "token": chunk_text,
                        "partial": True
                    }
                    yield f"data: {json.dumps(data)}\n\n"
                    time.sleep(0.01)  # Smooth streaming
            
            # Send completion signal
            data = {
                "token": "",
                "partial": False,
                "complete": True
            }
            yield f"data: {json.dumps(data)}\n\n"
            
        except Exception as e:
            error_data = {
                "error": str(e),
                "complete": True
            }
            yield f"data: {json.dumps(error_data)}\n\n"

    response = StreamingHttpResponse(
        event_stream(),
        content_type="text/event-stream"
    )
    response['Cache-Control'] = 'no-cache'
    response['X-Accel-Buffering'] = 'no'
    return response

# def chat_api_view(request):
#     """Non-streaming API endpoint (fallback)"""
#     if request.method == 'POST':
#         user_message = request.POST.get("message", "").strip()
        
#         if not user_message:
#             return JsonResponse({"error": "No message provided"}, status=400)
        
#         try:
#             initial_state = {
#                 "message": [HumanMessage(content=user_message)]
#             }
            
#             result = chatbot.invoke(initial_state, config=CONFIG)
#             response_text = result["message"][-1].content
            
#             return JsonResponse({
#                 "response": response_text,
#                 "status": "success"
#             })
            
#         except Exception as e:
#             return JsonResponse({
#                 "error": str(e),
#                 "status": "error"
#             }, status=500)
    
#     return JsonResponse({"error": "Invalid request method"}, status=405)