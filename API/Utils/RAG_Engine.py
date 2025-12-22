import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from settings import auto_config as cfg
from API.Utils.Databases.SQL_Database import DATABASE


class RAG:
    def __init__(self):
        try:
            self.model = SentenceTransformer(cfg.RAG_MODEL_PATH)
            self.enabled = True
        except Exception as e:
            print(f"[RAG] Disabled (model load failed): {e}")
            self.model = None
            self.enabled = False
        # Cache (performance): construir 1x e reutilizar
        self._documents = None
        self._index = None
        self._doc_embeddings = None

    def _get_documents_from_knowledge_source(self) -> np.ndarray:
        docs = DATABASE.get_all_knowledge_base()
        return np.array(docs, dtype=object)

    def _ensure_index(self):
        """Cria embeddings + index 1x por arranque do servidor (ou quando KB estiver vazia)."""
        if self._index is not None and self._documents is not None:
            return

        self._documents = self._get_documents_from_knowledge_source()

        if self._documents is None or len(self._documents) == 0:
            self._index = None
            self._doc_embeddings = None
            return

        # Embeddings (float32) + normalização -> usar cosine similarity via inner product
        doc_emb = self.model.encode(
            list(self._documents),
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype("float32")

        dim = doc_emb.shape[1]
        index = faiss.IndexFlatIP(dim)  # cosine similarity (com embeddings normalizados)

        index.add(doc_emb)

        self._doc_embeddings = doc_emb
        self._index = index

    def _search(self, query: str, k: int) -> str:
        self._ensure_index()
        if self._index is None or self._documents is None or len(self._documents) == 0:
            return ""

        # k tem de ser <= nº docs
        k = int(k)
        if k <= 0:
            return ""
        k = min(k, len(self._documents))

        q_emb = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype("float32")

        scores, indices = self._index.search(q_emb, k)

        # Como estamos a usar cosine similarity (IP com embeddings normalizados):
        # scores mais alto = mais parecido.
        # Se quiseres um limiar, define no cfg algo como MIN_RAG_SCORE = 0.3
        min_score = getattr(cfg, "MIN_RAG_SCORE", None)
        if min_score is not None:
            if scores[0][0] < float(min_score):
                return ""

        relevant_docs = [self._documents[i] for i in indices[0] if i != -1]
        return "\n---\n".join([str(d) for d in relevant_docs])

    async def find_contexts(self, query: str, k: int) -> str:
        if not self.enabled or self.model is None:
            return ""   # <- importante
        
        ctx = self._search(query, k)
        
        # Se não achou contexto, devolve vazio (para o controller decidir)
        if not ctx.strip():
            return ""

        # Prompt “RAG-only”: força a responder só com o contexto
        return (
            "You are an institutional assistant for IPB/CeDRI.\n"
            "You MUST answer ONLY using the provided CONTEXT.\n"
            "Treat CONTEXT as the single source of truth.\n"
            "Do NOT use external knowledge.\n"
            "If the answer is not explicitly in CONTEXT, reply exactly:\n"
            "\"I don't know based on the provided knowledge base.\"\n\n"
            f"CONTEXT:\n{ctx}\n\n"
            f"QUESTION:\n{query}\n\n"
            "ANSWER (write the answer directly, without mentioning the context):\n"
            "Do not say 'based on the context'." 
               
        )


RAG_ENGINE = RAG()
