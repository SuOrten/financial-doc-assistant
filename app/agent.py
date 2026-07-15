from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from typing import TypedDict, List
from app.ingest import get_vectorstore
import os
from dotenv import load_dotenv

load_dotenv()

class AgentState(TypedDict):
    question: str
    documents: List[str]
    answer: str

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
        SystemMessage(content="You are a financial document assistant. Answer questions based only on the provided context. If the answer is not in the context, say so clearly. Always cite which part of the document your answer comes from."),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {state['question']}")
    ]
    response = llm.invoke(messages)
    state["answer"] = response.content
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

def ask_question(question: str) -> str:
    result = agent.invoke({"question": question, "documents": [], "answer": ""})
    return result["answer"]
