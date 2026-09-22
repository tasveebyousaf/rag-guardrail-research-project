from datasets import load_dataset

# Loads only the first 200 examples from the validation split, not the whole
# ~90,000-example dataset, so this step stays fast while you're developing.
dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation[:200]")
print(f"Loaded {len(dataset)} examples")
print(dataset[0])

from llama_index.core import Document

documents = []
for example in dataset:
    titles = example["context"]["title"]
    sentences_lists = example["context"]["sentences"]
    for title, sentences in zip(titles, sentences_lists):
        text = " ".join(sentences)
        documents.append(Document(text=text, metadata={"title": title}))

print(f"Built {len(documents)} passage documents")

from llama_index.core import VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

index = VectorStoreIndex.from_documents(documents)
index.storage_context.persist(persist_dir="data/index")