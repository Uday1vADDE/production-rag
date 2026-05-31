import json
import os
import sys
sys.path.append("src")

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from pipeline import get_answers
from retrieval import retrieve
from ingest import process_pdf

load_dotenv()

eval_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)

def score_faithfulness(question, answer, chunks):
    context = "\n\n".join(chunks)
    prompt = f"""Given these retrieved chunks:
{context}

And this answer:
{answer}

Rate how faithful this answer is to the retrieved chunks on a scale of 0 to 1.
0 = answer contains claims not supported by chunks
1 = every claim in answer is supported by chunks

Return only a number between 0 and 1. Nothing else."""
    response = eval_llm.invoke([HumanMessage(content=prompt)])
    try:
        return float(response.content.strip())
    except:
        return 0.0

def score_answer_relevancy(question, answer):
    prompt = f"""Given this question:
{question}

And this answer:
{answer}

Rate how relevant this answer is to the question on a scale of 0 to 1.
0 = answer does not address the question at all
1 = answer directly and completely addresses the question

Return only a number between 0 and 1. Nothing else."""
    response = eval_llm.invoke([HumanMessage(content=prompt)])
    try:
        return float(response.content.strip())
    except:
        return 0.0

def score_context_precision(question, chunks):
    context = "\n\n".join(chunks)
    prompt = f"""Given this question:
{question}

And these retrieved chunks:
{context}

Rate how relevant these chunks are for answering the question on a scale of 0 to 1.
0 = chunks are completely irrelevant to the question
1 = all chunks are highly relevant to the question

Return only a number between 0 and 1. Nothing else."""
    response = eval_llm.invoke([HumanMessage(content=prompt)])
    try:
        return float(response.content.strip())
    except:
        return 0.0

def run_evaluation(pdf_path, dataset_path="evaluation/golden_dataset.json"):
    print("Loading golden dataset...")
    with open(dataset_path, "r") as f:
        golden_data = json.load(f)

    process_pdf(pdf_path)

    faithfulness_scores = []
    relevancy_scores = []
    precision_scores = []

    print(f"\nRunning evaluation on {len(golden_data)} questions...")
    print("="*50)

    for i, item in enumerate(golden_data):
        print(f"\nQuestion {i+1}/{len(golden_data)}: {item['question']}")

        answer = get_answers(pdf_path, item["question"])
        chunks = retrieve(pdf_path, item["question"], k=10, top_k=5)

        f_score = score_faithfulness(item["question"], answer, chunks)
        r_score = score_answer_relevancy(item["question"], answer)
        p_score = score_context_precision(item["question"], chunks)

        faithfulness_scores.append(f_score)
        relevancy_scores.append(r_score)
        precision_scores.append(p_score)

        print(f"  Faithfulness:      {f_score:.2f}")
        print(f"  Answer Relevancy:  {r_score:.2f}")
        print(f"  Context Precision: {p_score:.2f}")

    avg_faithfulness = sum(faithfulness_scores) / len(faithfulness_scores)
    avg_relevancy = sum(relevancy_scores) / len(relevancy_scores)
    avg_precision = sum(precision_scores) / len(precision_scores)

    print("\n" + "="*50)
    print("FINAL EVALUATION RESULTS")
    print("="*50)
    print(f"Faithfulness:      {avg_faithfulness:.3f}")
    print(f"Answer Relevancy:  {avg_relevancy:.3f}")
    print(f"Context Precision: {avg_precision:.3f}")
    print(f"Overall Score:     {(avg_faithfulness + avg_relevancy + avg_precision) / 3:.3f}")

    # Save results
    results = {
        "faithfulness": avg_faithfulness,
        "answer_relevancy": avg_relevancy,
        "context_precision": avg_precision,
        "overall": (avg_faithfulness + avg_relevancy + avg_precision) / 3
    }

    with open("evaluation/results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nResults saved to evaluation/results.json")
    return results

if __name__ == "__main__":
    run_evaluation("data/mcp_guide.pdf")