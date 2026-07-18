from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from typing import TypedDict, List
from app.ingest import get_vectorstore
import os
from dotenv import load_dotenv

load_dotenv()

class AgentState(TypedDict):
    question: str
    documents: List[str]
    answer: str
    chat_history: List[dict]

def retrieve(state: AgentState) -> AgentState:
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    docs = retriever.invoke(state["question"])
    state["documents"] = [doc.page_content for doc in docs]
    return state

def generate(state: AgentState) -> AgentState:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    context = "\n\n".join(state["documents"])

    messages = [
        SystemMessage(content="""You are a financial document assistant. 
        Answer questions based only on the provided context. 
        If the answer is not in the context, say so clearly.
        Always cite which part of the document your answer comes from.""")
    ]

    for turn in state["chat_history"]:
        messages.append(HumanMessage(content=turn["question"]))
        messages.append(AIMessage(content=turn["answer"]))

    messages.append(HumanMessage(content=f"Context:\n{context}\n\nQuestion: {state['question']}"))

    response = llm.invoke(messages)
    state["answer"] = response.content

    state["chat_history"].append({
        "question": state["question"],
        "answer": state["answer"]
    })

    return state

def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    return graph.compile()

agent = build_graph()

chat_history = []

def ask_question(question: str) -> str:
    global chat_history
    result = agent.invoke({
        "question": question,
        "documents": [],
        "answer": "",
        "chat_history": chat_history
    })
    chat_history = result["chat_history"]
    return result["answer"]

def reset_history():
    global chat_history
    chat_history = []