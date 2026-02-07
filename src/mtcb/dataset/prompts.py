"""Prompt templates for dataset generation."""

from dataclasses import dataclass


@dataclass
class DatasetPromptTemplate:
    """Configurable prompt templates for QA generation and deduplication.

    Attributes:
        generation_template: Template for generating QA pairs from chunks.
        deduplication_template: Template for LLM-based deduplication.
        name: Optional name for the template preset.

    """

    generation_template: str
    deduplication_template: str
    name: str = "custom"

    @classmethod
    def default(cls) -> "DatasetPromptTemplate":
        """Default template for general-purpose QA generation."""
        return cls(
            name="default",
            generation_template="""Given the following text chunk, generate a question-answer pair.

Chunk:
{chunk_text}

{existing_questions_section}

Requirements:
1. Create a clear, specific question answerable from this chunk
2. Generate a UNIQUE question different from any previously generated
3. Provide a concise, accurate answer
4. For chunk_must_contain: Copy an EXACT text span from the chunk that supports the answer
   - This must be a verbatim substring from the chunk
   - Preserve ALL formatting: spaces, line breaks, punctuation exactly as shown
   - Do NOT paraphrase or modify the text

Respond with JSON containing: question, answer, chunk_must_contain""",
            deduplication_template="""Analyze these questions for semantic similarity.
Group questions that ask essentially the same thing, even if worded differently.

Questions:
{questions_json}

Return JSON with:
- "groups": list of groups, each containing indices of similar questions
- "unique_indices": list of one representative index from each group""",
        )

    @classmethod
    def strict(cls) -> "DatasetPromptTemplate":
        """Strict template requiring longer, more specific source text spans."""
        return cls(
            name="strict",
            generation_template="""Given the following text chunk, generate a detailed question-answer pair.

Chunk:
{chunk_text}

{existing_questions_section}

STRICT Requirements:
1. Create a specific, detailed question that requires understanding of the chunk
2. Generate a UNIQUE question completely different from any previously generated
3. Provide a comprehensive answer with supporting details
4. For chunk_must_contain:
   - Copy an EXACT text span of at least 50 characters from the chunk
   - This must be a verbatim substring that DIRECTLY supports the answer
   - Include the most relevant sentence or phrase
   - Preserve ALL formatting exactly as shown

Respond with JSON containing: question, answer, chunk_must_contain""",
            deduplication_template="""Strictly analyze these questions for any semantic overlap.
Mark questions as duplicates if they:
- Ask about the same concept or fact
- Would have overlapping answers
- Cover the same information from different angles

Questions:
{questions_json}

Return JSON with:
- "groups": list of groups with indices of semantically similar questions
- "unique_indices": list of one best representative index from each group
- "reasoning": brief explanation for each grouping decision""",
        )

    @classmethod
    def financial(cls) -> "DatasetPromptTemplate":
        """Template optimized for financial documents."""
        return cls(
            name="financial",
            generation_template="""Given the following financial document chunk, generate a question-answer pair.

Chunk:
{chunk_text}

{existing_questions_section}

Financial Document Requirements:
1. Create questions about specific financial data, metrics, or statements
2. Focus on: numbers, percentages, dates, comparisons, trends, or regulatory items
3. Generate a UNIQUE question different from previously generated ones
4. Provide precise answers including exact figures when available
5. For chunk_must_contain:
   - Copy the EXACT text containing the financial data referenced
   - Include relevant numbers, dates, or metric names
   - Preserve all formatting exactly

Respond with JSON containing: question, answer, chunk_must_contain""",
            deduplication_template="""Analyze these financial questions for semantic similarity.
Consider questions duplicates if they ask about the same:
- Financial metric or KPI
- Time period comparison
- Company or entity performance
- Regulatory or compliance item

Questions:
{questions_json}

Return JSON with:
- "groups": list of groups with indices of similar questions
- "unique_indices": list of one representative index from each group""",
        )

    @classmethod
    def technical(cls) -> "DatasetPromptTemplate":
        """Template optimized for technical documentation."""
        return cls(
            name="technical",
            generation_template="""Given the following technical documentation chunk, generate a question-answer pair.

Chunk:
{chunk_text}

{existing_questions_section}

Technical Documentation Requirements:
1. Create questions about APIs, configurations, procedures, or specifications
2. Focus on: function names, parameters, code examples, or step-by-step processes
3. Generate a UNIQUE question different from previously generated ones
4. Provide technically accurate answers with specific details
5. For chunk_must_contain:
   - Copy the EXACT text containing the technical information
   - Include code snippets, parameter names, or command examples if relevant
   - Preserve all formatting including indentation

Respond with JSON containing: question, answer, chunk_must_contain""",
            deduplication_template="""Analyze these technical questions for semantic similarity.
Consider questions duplicates if they ask about the same:
- API endpoint or function
- Configuration option or parameter
- Process or procedure step
- Technical concept or definition

Questions:
{questions_json}

Return JSON with:
- "groups": list of groups with indices of similar questions
- "unique_indices": list of one representative index from each group""",
        )

    def format_generation_prompt(
        self, chunk_text: str, existing_questions: list[str] | None = None
    ) -> str:
        """Format the generation prompt with chunk text and existing questions."""
        if existing_questions:
            questions_list = "\n".join([f"- {q}" for q in existing_questions])
            existing_section = (
                f"Previously generated questions (do NOT duplicate):\n{questions_list}"
            )
        else:
            existing_section = "No previous questions generated yet."

        return self.generation_template.format(
            chunk_text=chunk_text,
            existing_questions_section=existing_section,
        )

    def format_deduplication_prompt(self, questions: list[dict]) -> str:
        """Format the deduplication prompt with questions."""
        import json

        questions_json = json.dumps(
            [{"index": i, "question": q["question"]} for i, q in enumerate(questions)],
            indent=2,
        )
        return self.deduplication_template.format(questions_json=questions_json)
