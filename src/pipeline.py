import os
import yaml
import re
import time
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
from retrieval import retrieve
from langfuse import observe, get_client

load_dotenv()

# ── LLMs ──
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.1
)

rewrite_llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)


def promt_call():
    path = "prompts/rag_prompt.yaml"
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return data


def rewrite_query(query, chat_history):
    if not chat_history:
        return query

    needs_rewrite = any(word in query.lower() for word in
        ["he", "she", "it", "they", "this", "that", "these", "those", "his", "her", "its", "their"])

    if not needs_rewrite:
        return query

    history_text = "\n".join([
        f"{msg['role'].upper()}: {msg['content']}"
        for msg in chat_history[-4:]
    ])

    rewrite_prompt = f"""Given this conversation history:
{history_text}

Rewrite this follow-up question as a complete standalone question by replacing all pronouns with their actual referents.

Follow-up question: {query}
Standalone question:"""

    response = rewrite_llm.invoke([HumanMessage(content=rewrite_prompt)])
    rewritten = response.content.strip()
    print(f"Original: {query}")
    print(f"Rewritten: {rewritten}")
    return rewritten


def build_prompt(chunks, query):
    data = promt_call()
    context = "\n\n".join([f"Chunk {i+1}: {chunk}" for i, chunk in enumerate(chunks)])
    rules_text = "\n".join([f"- {rule}" for rule in data["rag_prompt"]["rules"]])
    system_message = data["rag_prompt"]["system"] + "\n\nRules:\n" + rules_text
    filled_template = data["rag_prompt"]["template"].format(context=context, question=query)
    return filled_template, system_message


def has_citations(response):
    matches = re.findall(r'\[Chunk \d+\]', response)
    return len(matches) > 0


@observe(name="rag-query")
def get_answers(pdf_path, query, chat_history=[]):
    langfuse = get_client()

    # Set trace input/metadata
    langfuse.set_current_trace_io(
        input={"query": query, "pdf": os.path.basename(pdf_path)}
    )

    # 1. Query rewrite
    rewritten_query = rewrite_query(query, chat_history)

    # 2. Retrieval
    start = time.time()
    top_chunks = retrieve(pdf_path, rewritten_query, k=10, top_k=5)
    retrieval_latency = round((time.time() - start) * 1000)

    # 3. Prompt + LLM
    filled_template, system_message = build_prompt(top_chunks, rewritten_query)
    messages = [
        SystemMessage(content=system_message),
        HumanMessage(content=filled_template)
    ]

    start = time.time()
    response = llm.invoke(messages)
    llm_latency = round((time.time() - start) * 1000)

    # 4. Citation check
    if not has_citations(response.content):
        final_answer = "I don't have enough information in the provided document to answer this question."
    else:
        final_answer = response.content

    # Set trace output
    langfuse.set_current_trace_io(
        output={"answer": final_answer}
    )
    langfuse.score_current_trace(
        name="retrieval_latency_ms",
        value=retrieval_latency
    )
    langfuse.score_current_trace(
        name="llm_latency_ms",
        value=llm_latency
    )

    langfuse.flush()
    return final_answer