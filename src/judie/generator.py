"""Dataset generator for RAG evaluation using LLMs and chunking."""

import random
from typing import Dict, List, Optional, Union

from chonkie import BaseChunker, BaseGenie, GeminiGenie, RecursiveChunker
from datasets import Dataset, DatasetDict
from pydantic import BaseModel
from tqdm import tqdm


class QuestionAnswer(BaseModel):
    """Schema for generated question-answer pairs."""
    question: str
    answer: str
    source_text: str


class DatasetGenerator:
    """
    Generates RAG evaluation datasets from documents using intelligent chunking and QA generation.
    
    Features:
    - Document-level input (single document or list of documents)
    - Configurable questions per document
    - Intelligent chunk sampling based on questions_per_document
    - Smart question distribution across selected chunks
    - Source text validation with exact matching
    - HuggingFace DatasetDict output format
    """
    
    QUESTION_GENERATION_PROMPT = """Given the following text chunk, generate a clear, answerable question based on the content.

Chunk:
{chunk_text}

{existing_questions_section}

Requirements:
- Create a question that can be answered using information from this chunk
- Generate a UNIQUE question that is different from any previously generated questions
- Provide a concise, accurate answer to the question
- For source_text: Copy the EXACT text span from the chunk character-for-character
- CRITICAL: Preserve ALL formatting - spaces, line breaks, tabs, punctuation exactly as shown
- Do NOT reformat, clean up, or modify the text in any way
- The source_text must be a perfect substring that exists in the original chunk"""

    def __init__(
        self,
        genie: Optional[BaseGenie] = None,
        chunker: Optional[BaseChunker] = None,
        show_progress_bar: bool = True,
    ):
        """
        Initialize the DatasetGenerator.
        
        Args:
            genie: BaseGenie instance for text generation (defaults to GeminiGenie with Flash 2.5)
            chunker: Chunker to use (defaults to RecursiveChunker)
            show_progress_bar: Whether to show progress bar during generation
        """
        self.genie = genie or GeminiGenie(model="gemini-2.5-flash")
        self.chunker = chunker or RecursiveChunker()
        self.show_progress_bar = show_progress_bar

    def generate(
        self,
        documents: Union[str, List[str]],
        questions_per_document: int = 10,
        max_retries: int = 3,
    ) -> DatasetDict:
        """
        Generate evaluation dataset from documents.
        
        Args:
            documents: Single document string or list of document strings
            questions_per_document: Number of questions to generate per document
            max_retries: Maximum retries for validation failures
            
        Returns:
            HuggingFace DatasetDict with two splits:
            - 'corpus': Dataset containing documents with document_ids
            - 'qa': Dataset containing questions, answers, and source text
        """
        if isinstance(documents, str):
            documents = [documents]
            
        all_examples = []
        total_questions = len(documents) * questions_per_document
        
        # Create progress bar if enabled
        progress_bar = None
        if self.show_progress_bar:
            progress_bar = tqdm(
                total=total_questions,
                desc="Generating QA pairs",
                unit="questions"
            )
        
        try:
            for doc_id, document in enumerate(documents):
                doc_examples = self._process_document(
                    document, doc_id, questions_per_document, max_retries, progress_bar
                )
                all_examples.extend(doc_examples)
        finally:
            if progress_bar:
                progress_bar.close()
        
        # Create corpus dataset with documents and their IDs
        corpus_data = []
        for doc_id, document in enumerate(documents):
            corpus_data.append({
                "document_id": doc_id,
                "document": document
            })
        
        corpus_dataset = Dataset.from_list(corpus_data)
        qa_dataset = Dataset.from_list(all_examples)
        
        return DatasetDict({
            "corpus": corpus_dataset,
            "qa": qa_dataset
        })

    def __call__(
        self,
        documents: Union[str, List[str]],
        questions_per_document: int = 10,
        max_retries: int = 3,
    ) -> DatasetDict:
        """Callable interface - delegates to generate()."""
        return self.generate(documents, questions_per_document, max_retries)

    def _process_document(
        self, document: str, doc_id: int, questions_per_doc: int, max_retries: int, progress_bar=None
    ) -> List[Dict]:
        """Process single document and generate specified number of questions."""
        chunks = self.chunker.chunk(document)
        
        if not chunks:
            return []
        
        # Intelligently sample chunks based on questions_per_doc
        sampled_chunks = self._sample_chunks_intelligently(chunks, questions_per_doc)
        
        question_distribution = self._distribute_questions_across_chunks(
            sampled_chunks, questions_per_doc
        )
        
        examples = []
        existing_questions = []  # Track questions generated so far
        
        for chunk_idx, num_questions in question_distribution.items():
            chunk = sampled_chunks[chunk_idx]
            
            for _ in range(num_questions):
                example = self._generate_question_with_source(chunk, doc_id, max_retries, existing_questions)
                if example:
                    examples.append(example)
                    existing_questions.append(example['question'])
                
                # Update progress bar (whether successful or not)
                if progress_bar:
                    progress_bar.update(1)
        
        return examples

    def _sample_chunks_intelligently(self, chunks: List, questions_per_doc: int) -> List:
        """
        Intelligently sample chunks based on questions_per_document.
        
        Strategy:
        - If questions_per_doc >= num_chunks: use all chunks
        - If questions_per_doc < num_chunks: sample exactly questions_per_doc chunks (1 question per chunk)
        - Prefer longer chunks for sampling
        """
        if not chunks:
            return []
        
        num_chunks = len(chunks)
        
        # If we have fewer or equal chunks than questions, use all chunks
        if questions_per_doc >= num_chunks:
            return chunks
        
        # If we have more chunks than questions, sample exactly questions_per_doc chunks
        # This ensures 1 question per chunk
        num_to_sample = questions_per_doc
        
        # Weight chunks by length (prefer longer chunks)
        chunk_weights = []
        for chunk in chunks:
            chunk_text = chunk.text if hasattr(chunk, 'text') else str(chunk)
            weight = len(chunk_text.split())  # Word count as weight
            chunk_weights.append(weight)
        
        # Weighted random sampling
        if sum(chunk_weights) == 0:
            return random.sample(chunks, num_to_sample)
        
        # Use weighted sampling to prefer longer chunks
        sampled_indices = random.choices(
            range(len(chunks)), 
            weights=chunk_weights, 
            k=num_to_sample
        )
        
        # Remove duplicates while preserving order
        seen = set()
        sampled_chunks = []
        for idx in sampled_indices:
            if idx not in seen:
                sampled_chunks.append(chunks[idx])
                seen.add(idx)
        
        # If we need more chunks due to duplicates, add remaining randomly
        if len(sampled_chunks) < num_to_sample:
            remaining_indices = [i for i in range(len(chunks)) if i not in seen]
            additional_needed = num_to_sample - len(sampled_chunks)
            if remaining_indices:
                additional_indices = random.sample(
                    remaining_indices, 
                    min(additional_needed, len(remaining_indices))
                )
                sampled_chunks.extend([chunks[i] for i in additional_indices])
        
        return sampled_chunks

    def _distribute_questions_across_chunks(
        self, chunks: List, total_questions: int
    ) -> Dict[int, int]:
        """
        Distribute questions across chunks based on sampling strategy.
        
        Args:
            chunks: List of chunks to distribute questions across
            total_questions: Total number of questions to generate
            
        Returns:
            Dict mapping chunk index to number of questions
        """
        if not chunks:
            return {}
        
        num_chunks = len(chunks)
        
        # If we have more chunks than questions, we sampled exactly total_questions chunks
        # So each chunk gets exactly 1 question
        if total_questions <= num_chunks:
            return {i: 1 for i in range(len(chunks))}
        
        # If we have fewer chunks than questions, distribute based on chunk length
        chunk_weights = []
        for chunk in chunks:
            chunk_text = chunk.text if hasattr(chunk, 'text') else str(chunk)
            weight = len(chunk_text.split())
            chunk_weights.append(weight)
        
        total_weight = sum(chunk_weights)
        if total_weight == 0:
            questions_per_chunk = total_questions // len(chunks)
            remainder = total_questions % len(chunks)
            distribution = {i: questions_per_chunk for i in range(len(chunks))}
            for i in range(remainder):
                distribution[i] += 1
            return distribution
        
        distribution = {}
        remaining_questions = total_questions
        
        for i, weight in enumerate(chunk_weights[:-1]):
            questions = max(1, int((weight / total_weight) * total_questions))
            distribution[i] = min(questions, remaining_questions)
            remaining_questions -= distribution[i]
        
        distribution[len(chunks) - 1] = max(1, remaining_questions)
        
        return distribution

    def _generate_question_with_source(
        self, chunk, doc_id: int, max_retries: int, existing_questions: List[str] = None
    ) -> Optional[Dict]:
        """Generate question with exact source text, with validation."""
        chunk_text = chunk.text if hasattr(chunk, 'text') else str(chunk)
        existing_questions = existing_questions or []
        
        # Format existing questions section
        if existing_questions:
            questions_list = "\n".join([f"- {q}" for q in existing_questions])
            existing_questions_section = f"Previously generated questions (do NOT duplicate these):\n{questions_list}"
        else:
            existing_questions_section = "No previous questions generated yet."
        
        for attempt in range(max_retries):
            try:
                prompt = self.QUESTION_GENERATION_PROMPT.format(
                    chunk_text=chunk_text,
                    existing_questions_section=existing_questions_section
                )
                qa_data = self.genie.generate_json(prompt, QuestionAnswer)
                
                # Validate required keys exist
                required_keys = ['question', 'answer', 'source_text']
                if not all(key in qa_data for key in required_keys):
                    continue
                
                if self._validate_source_match(qa_data['source_text'], chunk_text):
                    return {
                        'document_id': doc_id,
                        'question': qa_data['question'],
                        'source_text': qa_data['source_text'],
                        'answer': qa_data['answer'],
                    }
                    
            except Exception:
                continue
        
        return None

    def _validate_source_match(self, source_text: str, chunk_text: str) -> bool:
        """Validate that source text exists exactly in chunk with exact formatting."""
        return source_text in chunk_text
    
    def __repr__(self) -> str:
        return f"DatasetGenerator(genie={self.genie}, chunker={self.chunker}, show_progress_bar={self.show_progress_bar})"