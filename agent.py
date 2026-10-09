from langchain_ollama import ChatOllama
from langchain.agents import create_agent

from rag_tool import search_university_policies


model = ChatOllama(
    model="qwen3:0.6b",
    temperature=0
)


agent = create_agent(
    model=model,
    tools=[search_university_policies],
    system_prompt="""
    You are a university IT support assistant.

    Use the search_university_policies tool to find relevant
    university IT policy information before answering.

    Use ONLY the information returned by the tool.

    If the tool does not provide the requested fact (for example,
    a phone number or a specific contact address), you MUST say
    that the information is not available.

    Never invent a phone number, email address, or any other fact
    that is not present in the retrieved policies.

    Give short and factual answers.
    """
)


if __name__ == "__main__":
    question = (
        "What exact phone number should I call "
        "for the University IT Service Desk?"
    )

    result = agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": question
            }
        ]
    })

    for message in result["messages"]:
        print("\n--------------------")
        print(type(message).__name__)
        print(message)