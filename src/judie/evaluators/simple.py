"""Simple evaluator implementation."""

import time
from typing import Any, Dict, List, Union
from tqdm import tqdm
from chonkie import Chunk, AutoEmbeddings
from ..evaluators.base import BaseEvaluator
from ..store import SimpleVectorStore
from ..types import EvalResult


class SimpleEvaluator(BaseEvaluator):
    """Simple evaluator for retrieval-based question answering systems.
    
    This evaluator assesses the performance of a chunking and retrieval system by:
    1. Chunking a corpus of documents using a provided chunker
    2. Embedding the chunks using a specified embedding model
    3. Storing chunks and embeddings in a vector store
    4. For each question, retrieving the top-k most similar chunks
    5. Checking if any retrieved chunks contain the relevant passages
    6. Computing accuracy metrics based on successful retrievals
    
    The evaluator supports both single documents (strings) and collections of documents (lists).
    It uses cosine similarity for vector-based retrieval and provides progress tracking.
    """
    def __init__(self,
                 corpus: Union[str, List[str]],
                 questions: List[str],
                 relevant_passages: Union[str, List[str]],
                 chunker: Any,
                 embedding_model: Union[str, Any] = "model2vec://minishlab/potion-base-8M") -> None:
        """Initialize the Evaluator.

        Args:
            corpus (Union[str, List[str]]): The corpus to evaluate on.
            questions (List[str]): The questions to evaluate on.
            relevant_passages (Union[List[str], List[List[str]]]): The relevant passages for each question.
            chunker (BaseChunker): The chunker to use.
            embedding_model (Union[str, Any], optional): The embedding model to use. Defaults to "bm25".

        """
        self.corpus = corpus
        self.questions = questions
        self.relevant_passages = relevant_passages
        self.chunker = chunker
        if isinstance(embedding_model, str):
            self.embedding_model = AutoEmbeddings.get_embeddings(embedding_model)
        else:
            self.embedding_model = embedding_model

    def _chunk(self, corpus: Union[str, List[str]]) -> List[Chunk]:
        """Chunk the corpus.

        Args:
            corpus (Union[str, List[str]]): The corpus to chunk.

        Returns:
            List[Chunk]: The chunks of the corpus.

        """
        if isinstance(corpus, str):
            return self.chunker(corpus)
        else:
            chunks = []
            for text in corpus:
                chunks.extend(self.chunker(text))
            return chunks

    def evaluate(self, k: Union[int, List[int]] = 1, show_progress_bar: bool = True) -> EvalResult:
        """Evaluate the corpus.

        Args:
            k: Either a single k value or a list of k values to evaluate
            show_progress_bar: Whether to show progress bar

        Returns:
            EvalResult: The results of the evaluation with metrics and performance stats.

        """
        eval_start_time = time.time()
        
        # Normalize k to a list for uniform processing
        k_values = [k] if isinstance(k, int) else k
        max_k = max(k_values)
        
        # Calculate corpus size
        if isinstance(self.corpus, str):
            corpus_size_mb = len(self.corpus.encode('utf-8')) / (1024 * 1024)
        else:
            corpus_size_mb = sum(len(text.encode('utf-8')) for text in self.corpus) / (1024 * 1024)
        
        # Time the chunking process
        chunk_start_time = time.time()
        chunks = self._chunk(self.corpus)
        chunk_end_time = time.time()
        
        total_time_to_chunk = chunk_end_time - chunk_start_time
        chunk_speed_mb_per_sec = corpus_size_mb / total_time_to_chunk if total_time_to_chunk > 0 else 0
        
        # Create a SimpleVectorStore object
        svs = SimpleVectorStore()
        embeddings = self.embedding_model.embed_batch([chunk.text for chunk in chunks])
        svs.add_batch(embeddings, chunks)

        # Initialize results for each k value
        results_by_k = {k_val: 0 for k_val in k_values}

        # Add the progress bar here which is enabled by show_progress_bar
        progress_bar = tqdm(enumerate(self.questions), total=len(self.questions), disable=not show_progress_bar)

        # Go through the list of questions and check which ones are there
        for i, question in progress_bar:
            # Check how many of the questions get the correct answer
            qemb = self.embedding_model.embed(question)
            results = svs.query(qemb, k=max_k)

            # Get the relevant passages for this specific question
            current_relevant_passages = self.relevant_passages[i]
            if isinstance(current_relevant_passages, str):
                current_relevant_passages = [current_relevant_passages]
            
            # Check for each k value
            for k_val in k_values:
                # Get top k_val results
                top_k_results = results[:k_val]
                if any(any(passage in chunk.text for passage in current_relevant_passages) for _, _, chunk in top_k_results):
                    results_by_k[k_val] += 1

        eval_end_time = time.time()
        total_evaluation_time = eval_end_time - eval_start_time
        questions_per_second = len(self.questions) / total_evaluation_time if total_evaluation_time > 0 else 0
        
        # Build metrics dictionary
        metrics = {
            "recall": {k_val: results_by_k[k_val] / len(self.questions) for k_val in k_values}
        }
        
        # Build metadata
        metadata = {
            "chunker_type": type(self.chunker).__name__,
            "chunker_config": repr(self.chunker),
            "embedding_model": str(self.embedding_model) if hasattr(self.embedding_model, '__str__') else "Unknown",
            "embedding_model_name": getattr(self.embedding_model, 'model', 'Unknown'),
            "total_questions": len(self.questions),
            "total_correct": results_by_k,
        }
        
        return EvalResult(
            metrics=metrics,
            metadata=metadata,
            total_corpus_size_mb=corpus_size_mb,
            total_time_to_chunk=total_time_to_chunk,
            chunk_speed_mb_per_sec=chunk_speed_mb_per_sec,
            total_chunks_created=len(chunks),
            total_evaluation_time=total_evaluation_time,
            questions_per_second=questions_per_second,
        )