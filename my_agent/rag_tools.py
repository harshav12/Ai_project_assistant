def search_knowledge_base_tool(query: str, user_id: str = None):
    """Search project issues using the local RAG index."""
    from my_agent.rag2_new import search_issues

    matches = search_issues(query=query, n_results=3)

    return [
        {
            "id": match["id"],
            "issue_id": match["metadata"].get("issue_id"),
            "text": match["text"],
            "source": "issues.json",
            "document_scope": "knowledge_base",
            "chunk_index": match["metadata"].get("issue_id", match["id"]),
            "distance": match["distance"],
            "metadata": match["metadata"],
        }
        for match in matches
    ]