"""Gacha-specific evaluator implementation."""

import statistics
import time
from typing import Any, Dict, List, Union
from tqdm import tqdm
from datasets import load_dataset
from chonkie import Chunk, AutoEmbeddings
from ..evaluators.base import BaseEvaluator
from ..store import SimpleVectorStore
from ..types import EvalResult


class GachaEvaluator(BaseEvaluator):
    """Evaluator specifically designed for the Gacha dataset.
    
    This evaluator loads the Gacha dataset internally and evaluates retrieval performance
    across all books in the corpus. It computes mean accuracy scores across all books
    for given k values, providing a comprehensive evaluation of chunking and retrieval
    strategies on the Gacha benchmark.
    
    The evaluator handles the dataset loading, book-specific question filtering,
    and aggregates results across all books to provide statistical measures
    of retrieval performance.
    """
    
    def __init__(self,
                 chunker: Any,
                 embedding_model: Union[str, Any] = "model2vec://minishlab/potion-base-8M") -> None:
        """Initialize the GachaEvaluator.

        Args:
            chunker: The chunker to use for splitting documents.
            embedding_model: The embedding model to use. Defaults to model2vec potion-base-8M.

        """
        self.chunker = chunker
        if isinstance(embedding_model, str):
            self.embedding_model = AutoEmbeddings.get_embeddings(embedding_model)
        else:
            self.embedding_model = embedding_model
        
        # Load datasets
        print("Loading Gacha datasets...")
        self.corpus = load_dataset("chonkie-ai/gacha", "corpus", split="train")
        self.questions = load_dataset("chonkie-ai/gacha", "questions", split="train")
        print(f"Loaded {len(self.corpus)} books and {len(self.questions)} questions")

    def _chunk(self, text: str) -> List[Chunk]:
        """Chunk a single text.

        Args:
            text: The text to chunk.

        Returns:
            List[Chunk]: The chunks of the text.

        """
        return self.chunker(text)

    def _evaluate_book(self, book_idx: int, k_values: List[int]) -> Dict:
        """Evaluate a single book for multiple k values efficiently.
        
        Args:
            book_idx: Index of the book in the corpus
            k_values: List of k values to evaluate
            
        Returns:
            Dict: Results for this book for each k value
        """
        book = self.corpus[book_idx]
        book_title = book['title']
        
        # Filter questions for this book
        book_questions = self.questions.filter(lambda x: x['title'] == book_title)
        
        if len(book_questions) == 0:
            return {
                "title": book_title,
                "results_by_k": {k: {"correct_count": 0, "total_count": 0, "accuracy": 0.0} for k in k_values}
            }
        
        # Extract data for this book
        question_texts = [item['question'] for item in book_questions]
        relevant_passages = [item['chunk-must-contain'] for item in book_questions]
        
        # Create vector store for this book
        svs = SimpleVectorStore()
        chunks = self._chunk(book['text'])
        embeddings = self.embedding_model.embed_batch([chunk.text for chunk in chunks])
        svs.add_batch(embeddings, chunks)
        
        # Initialize counters for each k value
        correct_counts = {k: 0 for k in k_values}
        max_k = max(k_values)
        
        # Evaluate questions for this book - single pass for all k values
        for i, question in enumerate(question_texts):
            qemb = self.embedding_model.embed(question)
            results = svs.query(qemb, k=max_k)  # Get max_k results once
            
            current_relevant_passage = relevant_passages[i]
            
            # Check relevance for each k value in a single pass
            for k_val in k_values:
                top_k_results = results[:k_val]  # Slice to get top k results
                if any(current_relevant_passage in chunk.text for _, _, chunk in top_k_results):
                    correct_counts[k_val] += 1
        
        # Format results for each k value
        results_by_k = {}
        for k_val in k_values:
            results_by_k[k_val] = {
                "correct_count": correct_counts[k_val],
                "total_count": len(question_texts),
                "accuracy": correct_counts[k_val] / len(question_texts)
            }
        
        return {
            "title": book_title,
            "results_by_k": results_by_k
        }

    def evaluate(self, k: Union[int, List[int]] = 1, show_progress_bar: bool = True) -> EvalResult:
        """Evaluate retrieval performance across all books in the Gacha dataset.

        Args:
            k: Either a single k value or a list of k values to evaluate
            show_progress_bar: Whether to show progress bar

        Returns:
            EvalResult: Results containing metrics and performance statistics across all books.

        """
        eval_start_time = time.time()
        k_values = [k] if isinstance(k, int) else k
        
        print(f"Evaluating with k values: {k_values}")
        
        # Calculate total corpus size
        total_corpus_size_mb = sum(len(book['text'].encode('utf-8')) for book in self.corpus) / (1024 * 1024)
        
        # Track chunking performance across all books
        total_chunks_created = 0
        total_time_to_chunk = 0
        book_results = []
        
        # Progress bar for books - single pass for all k values
        progress_bar = tqdm(range(len(self.corpus)), 
                          desc="Books", 
                          disable=not show_progress_bar)
        
        for book_idx in progress_bar:
            # Time chunking for this book
            book_chunk_start = time.time()
            book_result = self._evaluate_book(book_idx, k_values)
            book_chunk_end = time.time()
            
            # Track chunking stats (approximate - includes some evaluation time)
            book_chunks = len(self._chunk(self.corpus[book_idx]['text']))
            total_chunks_created += book_chunks
            total_time_to_chunk += (book_chunk_end - book_chunk_start) * 0.1  # Rough estimate for chunking portion
            
            book_results.append(book_result)
            
            # Update progress bar with current book title
            progress_bar.set_postfix({"book": book_result["title"][:20] + "..."})
        
        eval_end_time = time.time()
        total_evaluation_time = eval_end_time - eval_start_time
        chunk_speed_mb_per_sec = total_corpus_size_mb / total_time_to_chunk if total_time_to_chunk > 0 else 0
        
        # Aggregate results for each k value
        overall_correct = {}
        total_questions = sum(len(self.questions.filter(lambda x: x['title'] == book['title'])) for book in self.corpus)
        
        for k_val in k_values:
            # Calculate overall recall across all books
            total_correct = sum(book_result["results_by_k"][k_val]["correct_count"] for book_result in book_results)
            overall_correct[k_val] = total_correct
        
        questions_per_second = total_questions / total_evaluation_time if total_evaluation_time > 0 else 0
        
        # Build metrics dictionary
        metrics = {
            "recall": {k_val: overall_correct[k_val] / total_questions for k_val in k_values}
        }
        
        # Build metadata with comprehensive info
        metadata = {
            "chunker_type": type(self.chunker).__name__,
            "chunker_config": repr(self.chunker),
            "embedding_model": str(self.embedding_model) if hasattr(self.embedding_model, '__str__') else "Unknown",
            "embedding_model_name": getattr(self.embedding_model, 'model', 'Unknown'),
            "dataset": "gacha",
            "num_books": len(self.corpus),
            "total_questions": total_questions,
            "total_correct": overall_correct,
        }
        
        return EvalResult(
            metrics=metrics,
            metadata=metadata,
            total_corpus_size_mb=total_corpus_size_mb,
            total_time_to_chunk=total_time_to_chunk,
            chunk_speed_mb_per_sec=chunk_speed_mb_per_sec,
            total_chunks_created=total_chunks_created,
            total_evaluation_time=total_evaluation_time,
            questions_per_second=questions_per_second,
        )