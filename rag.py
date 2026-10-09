from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

from policy_loader import load_policy_documents


documents = load_policy_documents()

embeddings = OllamaEmbeddings(model="nomic-embed-text")

vector_store = Chroma(
    collection_name="university_it_policies",
    embedding_function=embeddings
)

vector_store.add_documents(documents)


def search_policies(question, k=3):
    results = vector_store.similarity_search(question, k=k)
    return results