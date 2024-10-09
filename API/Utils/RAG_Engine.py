import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from settings import auto_config as cfg
from API.Utils.Databases.SQL_Database import DATABASE

class RAG:
    def __init__(self):
        self.model = SentenceTransformer(cfg.RAG_MODEL_PATH)

    def _get_documents_from_knowledge_source(self) -> np.ndarray:
        documents = DATABASE.get_all_knowledge_base()
        return np.array(documents)

    def _exec_document_embedding(self, documents, query):
        return self.model.encode(documents), self.model.encode([query])

    def _build_faiss_index(self, document_embeddings):
        index = faiss.IndexFlatL2(document_embeddings.shape[1])
        index.add(np.array(document_embeddings))
        return index

    def _search_faiss_index(self, index, query_embedding, documents, k: int = 2):
        distances, indices = index.search(query_embedding, k)
        print(distances)
        if distances[0][0] > cfg.MAX_RAG_DISTANCE: return ""
        relevant_docs = [documents[i] for i in indices[0]]
        return " ".join(relevant_docs)

    async def find_contexts(self, query: str, k: int) -> str:
        documents = self._get_documents_from_knowledge_source()
        document_embeddings, query_embedding = self._exec_document_embedding(documents, query)
        index = self._build_faiss_index(document_embeddings)
        retrieved_context = self._search_faiss_index(index, query_embedding, documents, k)
        if retrieved_context == "": return query
        return f"Using this statement: {retrieved_context} Answer this question: {query} Answer:"

RAG_ENGINE = RAG()