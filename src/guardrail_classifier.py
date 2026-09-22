from transformers import pipeline
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Same rule as every other script that opens the index: the embed model
# has to be set again here, in this process, before load_index_from_storage.
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

storage_context = StorageContext.from_defaults(persist_dir="data/index")
index = load_index_from_storage(storage_context)
retriever = index.as_retriever(similarity_top_k=4)

# top_k=None returns a score for every label instead of collapsing to
# just the top one -- you need the raw probability, not just the verdict.
classifier = pipeline(
    "text-classification",
    model="protectai/deberta-v3-base-prompt-injection-v2",
    top_k=None,
)

passages = retriever.retrieve("Were Scott Derrickson and Ed Wood of the same nationality?")

for p in passages[:4]:
    scores = classifier(p.text[:512])[0]
    print(p.text[:100])
    for s in scores:
        print(f"  {s['label']}: {s['score']:.4f}")
    print()