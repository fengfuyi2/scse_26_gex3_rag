from langchain_core.tools import tool

from rag import search_policies


@tool
def search_university_policies(question: str) -> str:
    """
    Search the university IT policy documents for information
    relevant to the given question. Use this tool whenever the
    user asks about university IT rules, policies, or procedures.
    """

    results = search_policies(question, k=3)

    if not results:
        return "No relevant policy information found."

    output = ""

    for doc in results:
        output += f"[{doc.metadata['policy_id']} | {doc.metadata['topic']}]\n"
        output += doc.page_content
        output += "\n\n"

    return output