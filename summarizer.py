"""
HybridSummarizer: Text processing + TextRank extractive + optional abstractive (OpenAI or transformers).
Designed to be robust and provide a readable summary even when heavy models are unavailable.
"""

import os
from typing import List
import nltk
from nltk.tokenize import sent_tokenize
import numpy as np
from sentence_transformers import SentenceTransformer
import networkx as nx
import math
import logging

# Try optional imports
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

# Ensure NLTK punkt is available
nltk.download("punkt", quiet=True)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class HybridSummarizer:
    def __init__(self, embedding_model_name="all-MiniLM-L6-v2", abstractive_model_name="facebook/bart-large-cnn"):
        # Sentence transformer for embedding
        self.embedding_model_name = embedding_model_name
        try:
            self.embedder = SentenceTransformer(self.embedding_model_name)
        except Exception as e:
            logger.warning(f"SentenceTransformer load failed: {e}")
            self.embedder = None

        # Abstractive model name
        self.abstractive_model_name = abstractive_model_name
        self.abstractive_pipeline = None
        self._maybe_load_abstractive_pipeline()

        # Check for OpenAI key
        self.openai_key = os.environ.get("OPENAI_API_KEY")
        if self.openai_key:
            try:
                import openai
                openai.api_key = self.openai_key
                self.openai = openai
                logger.info("OpenAI key found and configured.")
            except Exception as e:
                logger.warning(f"OpenAI import/config failed: {e}")
                self.openai = None
        else:
            self.openai = None

    def _maybe_load_abstractive_pipeline(self):
        # Load transformers pipeline if available and resources permit
        if not TRANSFORMERS_AVAILABLE:
            logger.info("Transformers not available; abstractive pipeline will not be used.")
            return
        try:
            # instantiate lazily to avoid long startup delay; here we prepare pipeline
            self.abstractive_pipeline = pipeline("summarization", model=self.abstractive_model_name)
            logger.info(f"Loaded abstractive pipeline: {self.abstractive_model_name}")
        except Exception as e:
            logger.warning(f"Could not load transformers summarization pipeline: {e}")
            self.abstractive_pipeline = None

    @staticmethod
    def _preprocess(text: str) -> List[str]:
        # Basic cleaning + sentence tokenization
        # Keep punctuation because medical terms often include hyphens and parentheses
        text = text.replace("\r", " ").replace("\n", " ").strip()
        # collapse multiple spaces
        text = " ".join(text.split())
        sentences = sent_tokenize(text)
        # remove extremely short sentences
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        return sentences

    def _build_similarity_graph(self, sentences: List[str]):
        n = len(sentences)
        if n == 0:
            return nx.Graph()
        # Use sentence-transformers embeddings if available
        if self.embedder:
            try:
                embeddings = self.embedder.encode(sentences, convert_to_numpy=True, normalize_embeddings=True)
            except Exception as e:
                logger.warning(f"Embedding failure: {e}. Falling back to char-based sim.")
                embeddings = None
        else:
            embeddings = None

        G = nx.Graph()
        for i in range(n):
            G.add_node(i)

        # Create edge weights (cosine similarity if embeddings available, else simple normalized overlap)
        for i in range(n):
            for j in range(i + 1, n):
                if embeddings is not None:
                    # cosine sim: dot product of normalized vectors
                    weight = float(np.dot(embeddings[i], embeddings[j]))
                    # numerical stability clamp
                    weight = max(0.0, weight)
                else:
                    # cheap fallback: normalized common token count
                    si = set(sentences[i].lower().split())
                    sj = set(sentences[j].lower().split())
                    if len(si) + len(sj) == 0:
                        weight = 0.0
                    else:
                        weight = float(len(si.intersection(sj))) / (math.log(len(si) + 1) + math.log(len(sj) + 1))
                if weight > 0:
                    G.add_edge(i, j, weight=weight)
        return G

    def textrank_extract(self, text: str, top_k: int = 5) -> str:
        sentences = self._preprocess(text)
        if not sentences:
            return ""

        G = self._build_similarity_graph(sentences)
        # PageRank on weighted graph
        try:
            scores = nx.pagerank(G, weight="weight")
        except Exception:
            # fallback to degree-based ranking
            scores = {n: G.degree(n) for n in G.nodes}
        # rank and choose top_k sentences in original order
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_indices = sorted([idx for idx, _ in ranked[:min(top_k, len(ranked))]])
        summary = " ".join([sentences[i] for i in top_indices])
        return summary

    def abstractive_transformers(self, text: str, max_length=130, min_length=30) -> str:
        if not self.abstractive_pipeline:
            raise RuntimeError("Transformers abstractive pipeline not available.")
        # Many summarization models have token limits. Shorten if needed.
        # Provide the text directly; pipeline handles chunking poorly, so we send short extractive text.
        out = self.abstractive_pipeline(text, max_length=max_length, min_length=min_length, do_sample=False)
        if isinstance(out, list) and len(out) > 0 and "summary_text" in out[0]:
            return out[0]["summary_text"].strip()
        elif isinstance(out, list) and len(out) > 0:
            # older transformers return text-key
            return out[0].get("summary_text", str(out[0])).strip()
        else:
            return str(out).strip()

    def abstractive_openai(self, text: str, max_tokens=200) -> str:
        if not self.openai:
            raise RuntimeError("OpenAI client not configured.")
        # Use chat completions if available; otherwise fallback.
        prompt = (
            "You are a helpful assistant specialized in medical text summarization. "
            "Provide a concise, clinically accurate summary of the following medical document. "
            "Keep the summary factual and avoid hallucinations.\n\nDocument:\n"
            f"{text}\n\nSummary:"
        )
        try:
            # Use the modern ChatCompletions if available
            if hasattr(self.openai, "ChatCompletion"):
                resp = self.openai.ChatCompletion.create(
                    model="gpt-4o-mini" if False else "gpt-4o-mini" if False else "gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=0.0,
                )
                # extract
                if "choices" in resp and len(resp["choices"]) > 0:
                    return resp["choices"][0]["message"]["content"].strip()
            # fallback to completions
            resp = self.openai.Completion.create(
                engine="text-davinci-003",
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=0.0,
            )
            if "choices" in resp and len(resp["choices"]) > 0:
                return resp["choices"][0]["text"].strip()
        except Exception as e:
            logger.warning(f"OpenAI summarization failed: {e}")
            raise

    def summarize(self, text: str, method: str = "hybrid") -> str:
        """
        method: 'extractive' | 'abstractive' | 'hybrid'
        hybrid: runs TextRank to get extractive seed then uses abstractive model if available
        """
        method = method.lower()
        if method not in ("extractive", "abstractive", "hybrid"):
            method = "hybrid"

        # Standard extractive baseline
        extractive = self.textrank_extract(text, top_k=6)
        if method == "extractive":
            return extractive

        # For abstractive-only: attempt to summarise full text via models (might fail for long docs)
        if method == "abstractive":
            # prefer OpenAI if configured
            if self.openai:
                try:
                    return self.abstractive_openai(text)
                except Exception:
                    # fallback to transformers pipeline
                    pass
            if self.abstractive_pipeline:
                try:
                    return self.abstractive_transformers(extractive if len(extractive) > 0 else text)
                except Exception as e:
                    logger.warning(f"Transformers abstractive failed: {e}")
                    raise RuntimeError("No abstractive summarizer available.")
            raise RuntimeError("No abstractive summarizer available.")

        # hybrid
        # 1) get extractive seeds
        if not extractive:
            return ""
        # 2) try OpenAI
        if self.openai:
            try:
                # limit size by passing extractive as context
                prompt_text = f"Summarize the following medical text concisely and precisely:\n\n{extractive}"
                return self.abstractive_openai(prompt_text)
            except Exception as e:
                logger.warning(f"OpenAI hybrid failed: {e}")

        # 3) try transformers pipeline
        if self.abstractive_pipeline:
            try:
                return self.abstractive_transformers(extractive)
            except Exception as e:
                logger.warning(f"Transformers hybrid failed: {e}")

        # 4) fallback to extractive
        logger.info("Falling back to extractive summary.")
        return extractive
