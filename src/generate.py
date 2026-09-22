import os
from dotenv import load_dotenv
from groq import Groq
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

load_dotenv()
client = Groq(api_key=os.environ["GROQ_API_KEY"])

# Same rule as query_test.py: this is a separate script run, so the embed
# model has to be set again here — Settings doesn't persist between runs.
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

storage_context = StorageContext.from_defaults(persist_dir="data/index")
index = load_index_from_storage(storage_context)
retriever = index.as_retriever(similarity_top_k=3)


def generate_answer(question: str) -> str:
    passages = retriever.retrieve(question)
    context = "\n\n".join(p.text for p in passages)
    prompt = (
        "Answer the question using only the context below. "
        "If the context doesn't contain the answer, say so.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    test_questions = [
        "Were Scott Derrickson and Ed Wood of the same nationality?",
        "What nationality was Scott Derrickson?",
    ]
    for q in test_questions:
        print(f"Q: {q}")
        print(f"A: {generate_answer(q)}\n")