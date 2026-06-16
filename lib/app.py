from __future__ import annotations

from flask import Flask, jsonify, request

from lib.ai_client import generate_response
from lib.company_documents import COMPANY_DOCUMENTS
from lib.rag_service import build_prompt, retrieve_context, source_metadata


def create_app():
    app = Flask(__name__)

    @app.get("/api/health")
    def health_check():
        return jsonify({"status": "ok"})

    @app.post("/api/ask")
    def ask_question():
        """Accept a query and return a source-backed generated answer.

        TODO:
        1. Read JSON request data safely.
        2. Validate that `query` is a non-empty string.
        3. Retrieve relevant context from COMPANY_DOCUMENTS.
        4. If no context is found, return a safe fallback with an empty sources list.
        5. Build a structured prompt from the selected context.
        6. Call generate_response(prompt).
        7. Return query, answer, and sources as JSON.
        8. If generate_response raises RuntimeError, return a 503 service error.
        """
        data = request.get_json(silent=True)

        if not data or "query" not in data:
            return jsonify({"error": "Request must include a 'query' field."}), 400

        query = data.get("query")

        if not isinstance(query, str):
            return jsonify({"error": "The 'query' must be a string."}), 400

        if not query.strip():
            return jsonify({"error": "The 'query' must not be blank."}), 400

        # Retrieve context
        matches = retrieve_context(query, COMPANY_DOCUMENTS)

        # If no context, return safe fallback without calling model
        if not matches:
            answer = (
                "I could not find relevant information in the approved company documents. "
                "The available documents do not contain enough information to answer this question."
            )
            return jsonify({"query": query, "answer": answer, "sources": []}), 200

        # Build prompt and call model
        prompt = build_prompt(query, matches)

        try:
            generated = generate_response(prompt)
        except RuntimeError as err:
            return (
                jsonify({"error": f"Model service error: {str(err)}"}),
                503,
            )

        sources = [source_metadata(m) for m in matches]

        return jsonify({"query": query, "answer": generated, "sources": sources}), 200

    return app


app = create_app()


if __name__ == "__main__":
    app.run(debug=True)
