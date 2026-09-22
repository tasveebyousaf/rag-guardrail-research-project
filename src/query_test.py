from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Must match the embed model used when the index was built in step 5 —
# without this, LlamaIndex falls back to its OpenAI default and errors
# out looking for an API key you don't have (and don't need).
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

storage_context = StorageContext.from_defaults(persist_dir="data/index")
index = load_index_from_storage(storage_context)

retriever = index.as_retriever(similarity_top_k=3)
results = retriever.retrieve("Were Scott Derrickson and Ed Wood of the same nationality?")

for r in results:
    print(round(r.score, 3), "|", r.text[:150])