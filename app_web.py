"""
Askforge — Web version.

Same RAG pipeline as app.py, served through a small Flask API + chat UI.
"""

import os
from pathlib import Path

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

from core import (
    RAGError,
    WEB_SEARCH_ENABLED,
    DOCS_DIR,
    SUPPORTED_DOCUMENT_EXTENSIONS,
    SUPPORTED_IMAGE_EXTENSIONS,
    build_or_load_vectorstore,
    add_document_to_vectorstore,
    answer_question,
    answer_image_question,
)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5 MB per upload, plenty for .txt

# Populated once at startup by initialize_documents(); None until then.
vectorstore = None
startup_error = None


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/ask", methods=["POST"])
def api_ask():
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    image_b64 = data.get("image_base64")
    image_mime = data.get("image_mime")

    if not question and not image_b64:
        return jsonify({"error": "Please enter a question."}), 400

    if startup_error:
        return jsonify({"error": startup_error}), 503

    try:
        if image_b64:
            if not image_mime:
                return jsonify({"error": "Missing image type."}), 400
            result = answer_image_question(question, image_b64, image_mime)
        else:
            if vectorstore is None and not WEB_SEARCH_ENABLED:
                return jsonify({"error": "No documents loaded and web search is disabled. Add files to docs/ and restart."}), 400
            result = answer_question(vectorstore, question)
    except RAGError as e:
        return jsonify({"error": str(e)}), 500
    except Exception:
        # Don't leak internal exception details (API errors, stack traces, etc.) to the client.
        app.logger.exception("Error answering question")
        return jsonify({"error": "Something went wrong while generating an answer. Please try again."}), 500

    return jsonify({"answer": result["answer"], "sources": result["sources"], "web_sources": result["web_sources"]})


@app.route("/api/upload", methods=["POST"])
def api_upload():
    global vectorstore

    if startup_error:
        return jsonify({"error": startup_error}), 503

    uploaded = request.files.get("file")
    if uploaded is None or uploaded.filename == "":
        return jsonify({"error": "No file was sent."}), 400

    filename = secure_filename(uploaded.filename)
    if not filename:
        return jsonify({"error": "That filename isn't valid."}), 400

    suffix = Path(filename).suffix.lower()
    if suffix in SUPPORTED_IMAGE_EXTENSIONS:
        return jsonify({"error": "Images aren't added to the knowledge base - attach one and ask about it directly instead."}), 400
    if suffix not in SUPPORTED_DOCUMENT_EXTENSIONS:
        allowed = ", ".join(sorted(SUPPORTED_DOCUMENT_EXTENSIONS))
        return jsonify({"error": f"'{suffix or filename}' isn't supported. Allowed: {allowed}"}), 400

    DOCS_DIR.mkdir(exist_ok=True)
    destination = DOCS_DIR / filename
    # Avoid clobbering an existing file with the same name.
    counter = 1
    while destination.exists():
        destination = DOCS_DIR / f"{Path(filename).stem}_{counter}{suffix}"
        counter += 1

    uploaded.save(destination)

    try:
        vectorstore = add_document_to_vectorstore(vectorstore, destination)
    except RAGError as e:
        destination.unlink(missing_ok=True)
        return jsonify({"error": str(e)}), 500
    except Exception:
        destination.unlink(missing_ok=True)
        app.logger.exception("Error adding uploaded document")
        return jsonify({"error": "Something went wrong while processing that file. Please try again."}), 500

    return jsonify({"filename": destination.name, "message": f"'{destination.name}' was added to your knowledge base."})


@app.route("/api/status", methods=["GET"])
def api_status():
    if startup_error:
        return jsonify({"loaded": False, "message": startup_error})

    if vectorstore is None:
        if WEB_SEARCH_ENABLED:
            return jsonify({"loaded": True, "message": "No documents loaded - answering from web search only"})
        return jsonify({"loaded": False, "message": "No documents loaded and web search is disabled"})

    if WEB_SEARCH_ENABLED:
        return jsonify({"loaded": True, "message": "Documents indexed - answering from documents + web search"})
    return jsonify({"loaded": True, "message": "Documents indexed and ready"})


def initialize_documents():
    """Build (or load) the vector store once when the server starts."""
    global vectorstore, startup_error
    try:
        vectorstore = build_or_load_vectorstore()
        if vectorstore is None:
            print("No documents found in docs/." + (" Continuing with web search only." if WEB_SEARCH_ENABLED else " Add files and restart."))
        else:
            print("Documents indexed and ready.")
    except RAGError as e:
        startup_error = str(e)
        print(f"Setup error: {e}")


# Runs at import time - critical for production servers like gunicorn, which
# import this module and use the `app` object directly without ever
# executing the `if __name__ == "__main__":` block below. Without this at
# module level, gunicorn deployments would silently never index any
# documents (no error either - just an empty knowledge base forever).
initialize_documents()


if __name__ == "__main__":
    debug_mode = os.getenv("FLASK_DEBUG", "false").lower() == "true"

    # Railway (and most hosts) assign a random port via the PORT env var and
    # expect the app to listen on 0.0.0.0, not 127.0.0.1 - localhost-only
    # binding is exactly why "Application failed to respond" shows up, since
    # their proxy can't reach a server that's only listening on loopback.
    # Locally (no PORT set), default back to 127.0.0.1:5000 for safety.
    platform_port = os.getenv("PORT")
    host = os.getenv("FLASK_HOST", "0.0.0.0" if platform_port else "127.0.0.1")
    port = int(os.getenv("FLASK_PORT", platform_port or "5000"))

    print(f"\nStarting web server on {host}:{port}")
    if vectorstore is None and not startup_error:
        print("Add documents to the docs/ folder to answer from your own content too.\n")
    else:
        print()
    app.run(debug=debug_mode, host=host, port=port)
