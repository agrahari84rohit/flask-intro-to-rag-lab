from __future__ import annotations

import re
from typing import Any

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "can",
    "do",
    "does",
    "for",
    "from",
    "get",
    "how",
    "i",
    "if",
    "in",
    "is",
    "it",
    "me",
    "my",
    "need",
    "of",
    "on",
    "or",
    "our",
    "should",
    "so",
    "the",
    "their",
    "to",
    "use",
    "what",
    "when",
    "where",
    "who",
    "why",
    "with",
    "you",
    "your",
}


def tokenize(text: str) -> set[str]:
    """Convert text into a set of searchable lowercase tokens.

    TODO:
    - Lowercase the text.
    - Extract word-like values.
    - Remove leading/trailing apostrophes.
    - Remove tokens with length <= 1.
    - Remove tokens in STOPWORDS.
    - Return a set of searchable terms.
    """
    if not text:
        return set()

    text = text.lower()
    # Extract words and numbers, keep apostrophes inside words then strip them
    raw_tokens = re.findall(r"[\w']+", text)
    tokens = set()
    for t in raw_tokens:
        # remove leading/trailing apostrophes
        t = t.strip("'\"")
        if len(t) <= 1:
            continue
        if t in STOPWORDS:
            continue
        tokens.add(t)

    return tokens


def document_search_text(document: dict[str, Any]) -> str:
    """Combine searchable document fields into one text value.

    TODO:
    Include title, category, tags, and text.
    """
    parts: list[str] = []
    title = document.get("title", "")
    category = document.get("category", "")
    tags = document.get("tags", [])
    text = document.get("text", "")

    parts.append(str(title))
    parts.append(str(category))
    if isinstance(tags, (list, tuple)):
        parts.append(" ".join(map(str, tags)))
    else:
        parts.append(str(tags))
    parts.append(str(text))

    return " ".join(parts)


def score_document(query: str, document: dict[str, Any]) -> dict[str, Any]:
    """Score a document using keyword overlap.

    TODO:
    - Tokenize the query.
    - Tokenize the combined searchable document text.
    - Tokenize the document title.
    - Find matched terms between query tokens and document tokens.
    - Add a small title boost: 0.5 for each query token found in the title.
    - Return a dictionary with keys: document, score, matched_terms.
    """
    query_tokens = tokenize(query)
    doc_text = document_search_text(document)
    doc_tokens = tokenize(doc_text)
    title_tokens = tokenize(document.get("title", ""))

    matched = sorted(query_tokens & doc_tokens)

    # base score: number of matched tokens
    score = float(len(matched))

    # title boost: 0.5 for each query token found in title
    title_matches = query_tokens & title_tokens
    score += 0.5 * len(title_matches)

    return {
        "document": document,
        "score": score,
        "matched_terms": matched,
    }


def retrieve_context(
    query: str,
    documents: list[dict[str, Any]],
    limit: int = 2,
    minimum_score: float = 1.0,
) -> list[dict[str, Any]]:
    """Select the most relevant documents for the query.

    TODO:
    - Score all documents.
    - Keep only matches with score >= minimum_score.
    - Sort by score from highest to lowest.
    - Return only the top `limit` matches.

    The selected context must depend on the user's query. Do not return the same
    hardcoded document for every request.
    """
    scored = [score_document(query, d) for d in documents]
    # filter by minimum_score
    filtered = [s for s in scored if s["score"] >= minimum_score]
    # sort by score desc
    filtered.sort(key=lambda x: x["score"], reverse=True)
    return filtered[:limit]


def format_context(context_matches: list[dict[str, Any]]) -> str:
    """Format retrieved documents into a context block for the prompt.

    TODO:
    - If no matches exist, return a short no-context message.
    - For each match, include Source ID, Title, Category, and Content.
    - Separate document blocks clearly.
    """
    if not context_matches:
        return "No relevant context found for this query."

    blocks: list[str] = []
    for match in context_matches:
        doc = match["document"]
        block = []
        block.append(f"Source ID: {doc.get('id', '')}")
        block.append(f"Title: {doc.get('title', '')}")
        block.append(f"Category: {doc.get('category', '')}")
        block.append("Content:")
        block.append(doc.get("text", ""))
        blocks.append("\n".join(block))

    return "\n\n---\n\n".join(blocks)


def build_prompt(query: str, context_matches: list[dict[str, Any]]) -> str:
    """Build a structured prompt with instructions, context, question, and requirements.

    TODO:
    The prompt should include these sections:
    - Instructions
    - Context
    - Question
    - Response requirements

    The prompt should tell the model to use only the provided context and avoid
    inventing unsupported details.
    """
    instructions = (
        "Instructions: You are an assistant that answers user questions using ONLY the provided context. "
        "Use only the provided context to answer the question. "
        "If the context does not contain the information needed, state that you do not have enough information. "
        "Do not invent details or make assumptions."
    )

    context_block = format_context(context_matches)

    question = f"Question: {query}"

    requirements = (
        "Response requirements: Provide a concise answer based only on the Context. "
        "Include a short list of source IDs used in the format 'Sources: id1, id2'. "
        "Do not include unrelated information."
    )

    prompt_parts = ["Instructions:", instructions, "", "Context:", context_block, "", question, "", "Response requirements:", requirements]
    return "\n".join(prompt_parts)


def source_metadata(match: dict[str, Any]) -> dict[str, str]:
    """Return source information that is safe to expose in the API response.

    TODO:
    Return only the document id and title.
    """
    doc = match.get("document", {})
    return {"id": doc.get("id", ""), "title": doc.get("title", "")}
