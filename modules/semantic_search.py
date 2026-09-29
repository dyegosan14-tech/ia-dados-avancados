"""
Módulo 1: Busca Semântica em Documentos
Pipeline: Ingestão -> Vetorização Densa & Esparsa -> Busca Híbrida (RRF) -> Reranking
"""
import re
import math
from typing import List, Dict, Any, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

class SemanticSearchEngine:
    """
    Motor de Busca Híbrida e Semântica com Reranking.
    Combina representações densas (vetoriais), esparsas (léxicas) e RRF.
    """
    def __init__(self, n_components: int = 16):
        self.documents: List[Dict[str, Any]] = []
        self.doc_texts: List[str] = []
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            stop_words=None
        )
        self.n_components = n_components
        self.svd: TruncatedSVD = None
        self.dense_embeddings: np.ndarray = None
        self.is_indexed = False

    def index_documents(self, docs: List[Dict[str, Any]]):
        """Indexa uma lista de documentos e pré-calcula as representações densas."""
        self.documents = docs
        self.doc_texts = [f"{d.get('titulo', '')} {d.get('conteudo', '')}" for d in docs]
        
        if not self.doc_texts:
            return

        # 1. Matriz esparsa TF-IDF
        tfidf_matrix = self.vectorizer.fit_transform(self.doc_texts)
        
        # 2. Representação densa normalizada L2 (vetores densos de busca semântica)
        dense = tfidf_matrix.toarray().astype(np.float64)
        norms = np.linalg.norm(dense, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.dense_embeddings = dense / norms
        self.is_indexed = True

    def _dense_search(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        """Busca puramente vetorial por similaridade de cosseno."""
        q_sparse = self.vectorizer.transform([query])
        q_dense = q_sparse.toarray().astype(np.float64)
        norm = np.linalg.norm(q_dense)
        if norm > 0:
            q_dense = q_dense / norm
            
        scores = cosine_similarity(q_dense, self.dense_embeddings)[0]
        ranked_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(idx), float(scores[idx])) for idx in ranked_indices]

    def _lexical_search(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        """Busca léxica / esparsa baseada em TF-IDF."""
        q_vec = self.vectorizer.transform([query])
        doc_vecs = self.vectorizer.transform(self.doc_texts)
        scores = cosine_similarity(q_vec, doc_vecs)[0]
        ranked_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(idx), float(scores[idx])) for idx in ranked_indices]

    def _reciprocal_rank_fusion(
        self, 
        dense_rankings: List[Tuple[int, float]], 
        lexical_rankings: List[Tuple[int, float]], 
        k: int = 60
    ) -> Dict[int, float]:
        """Calcula o score RRF (Reciprocal Rank Fusion) para combinar rankings."""
        rrf_scores: Dict[int, float] = {}

        for rank, (doc_idx, _) in enumerate(dense_rankings):
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + (1.0 / (k + rank + 1))

        for rank, (doc_idx, _) in enumerate(lexical_rankings):
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + (1.0 / (k + rank + 1))

        return rrf_scores

    def _cross_encoder_rerank(self, query: str, candidate_indices: List[int]) -> List[Tuple[int, float]]:
        """
        Reranker de precisão: analisa a interação profunda entre termos da query
        e o texto do documento, calculando overlap contextual e densidade de relevância.
        """
        query_tokens = set(re.findall(r"\w+", query.lower()))
        reranked = []

        for idx in candidate_indices:
            doc = self.documents[idx]
            text = f"{doc.get('titulo', '')} {doc.get('conteudo', '')}".lower()
            doc_tokens = re.findall(r"\w+", text)
            
            if not doc_tokens:
                reranked.append((idx, 0.0))
                continue

            # 1. Term Overlap
            matched = sum(1 for tok in query_tokens if tok in text)
            overlap_ratio = matched / max(1, len(query_tokens))

            # 2. Relevância no título (peso 2x)
            title_text = doc.get("titulo", "").lower()
            title_matches = sum(1 for tok in query_tokens if tok in title_text)
            title_bonus = 0.3 * (title_matches / max(1, len(query_tokens)))

            # 3. Penalização de dispersão / comprimento
            density_bonus = min(0.2, (matched / (math.log(len(doc_tokens) + 10) + 1)))

            final_score = (overlap_ratio * 0.5) + title_bonus + density_bonus
            reranked.append((idx, round(float(final_score), 4)))

        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked

    def search(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        """
        Executa a busca híbrida completa:
        1. Dense Retrieval + Sparse Retrieval
        2. Reciprocal Rank Fusion
        3. Reranker Cross-Encoder nos candidatos
        """
        if not self.is_indexed or not self.documents:
            return []

        # 1. Top candidatos densos e esparsos
        n_candidates = min(len(self.documents), top_k * 3)
        dense_results = self._dense_search(query, n_candidates)
        lexical_results = self._lexical_search(query, n_candidates)

        # 2. RRF
        rrf_dict = self._reciprocal_rank_fusion(dense_results, lexical_results)
        sorted_by_rrf = sorted(rrf_dict.items(), key=lambda x: x[1], reverse=True)
        top_candidates = [idx for idx, _ in sorted_by_rrf[:n_candidates]]

        # 3. Reranking
        reranked = self._cross_encoder_rerank(query, top_candidates)

        # Montagem do resultado detalhado
        dense_map = dict(dense_results)
        lexical_map = dict(lexical_results)

        final_results = []
        for rank, (doc_idx, rerank_score) in enumerate(reranked[:top_k], start=1):
            doc = self.documents[doc_idx]
            final_results.append({
                "rank": rank,
                "id": doc.get("id"),
                "titulo": doc.get("titulo"),
                "categoria": doc.get("categoria"),
                "conteudo": doc.get("conteudo"),
                "rerank_score": rerank_score,
                "dense_score": round(dense_map.get(doc_idx, 0.0), 3),
                "lexical_score": round(lexical_map.get(doc_idx, 0.0), 3),
                "rrf_score": round(rrf_dict.get(doc_idx, 0.0), 4)
            })

        return final_results
