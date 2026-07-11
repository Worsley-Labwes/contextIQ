"""
Askforge — CLI version.

Loads .txt files from docs/, embeds them into a local Chroma vector store,
and answers questions using an LLM (OpenAI or Google Gemini) grounded in
that context - plus live web search results when relevant.
"""

from core import RAGError, WEB_SEARCH_ENABLED, build_or_load_vectorstore, answer_question, DOCS_DIR


def main():
    vectorstore = None
    try:
        print("Loading documents and preparing the vector store (first run may take a moment)...")
        vectorstore = build_or_load_vectorstore()
    except RAGError as e:
        print(f"Setup error: {e}")
        return

    if vectorstore is None:
        if WEB_SEARCH_ENABLED:
            print("No .txt documents found in docs/ - continuing with web search only.")
        else:
            print("No .txt documents found in docs/, and web search is disabled. Add files and try again.")
            return

    print("Ready. Ask a question (type 'exit' to quit):\n")

    while True:
        try:
            question = input("Question> ").strip()
            if not question:
                continue
            if question.lower() in {"exit", "quit", "q"}:
                print("Goodbye.")
                break

            result = answer_question(vectorstore, question)
            print(f"\nAnswer:\n{result['answer']}\n")
            if result["sources"]:
                print(f"(Document sources: {', '.join(result['sources'])})")
            if result["web_sources"]:
                web_list = ", ".join(f"{w['title']} ({w['url']})" for w in result["web_sources"])
                print(f"(Web sources: {web_list})")
            print()
        except KeyboardInterrupt:
            print("\nGoodbye.")
            break
        except RAGError as e:
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    main()
