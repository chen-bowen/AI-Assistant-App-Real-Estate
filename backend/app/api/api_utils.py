import json

from llama_index.core.indices.vector_store import VectorStoreIndex
from llama_index.core.llms import ChatMessage

from app.api.schema import ChatMessages


def build_context_from_index(index: VectorStoreIndex, lastMessage: ChatMessage) -> str:
    """
    Builds the context from the given index and last message.

    Args:
        index (VectorStoreIndex): The index used for retrieval.
        lastMessage (ChatMessage): The last message in the conversation.

    Returns:
        str: The context built from the retrieved nodes.
    """

    # set the index as retriever
    retriever = index.as_retriever(similarity_top_k=2)

    # retrieve the context from the question
    nodes = retriever.retrieve(lastMessage.content)

    # build the context from the nodes
    context = ""
    for node in nodes:
        context = context + node.text + " "

    return context


def build_message_sequence(data: ChatMessages, context: str, lastMessage: ChatMessage) -> str:
    """
    Builds a message sequence from the given list of messages.

    Args:
        messages (list[ChatMessage]): The list of messages to build the sequence from.

    Returns:
        str: The message sequence built from the given messages.
    """

    # convert messages coming from the request to type ChatMessage
    messages = [
        ChatMessage(
            role=m.role,
            content=m.content,
        )
        for m in data.messages
    ]
    # create the user message with the context and question
    user_message = ChatMessage(role="user", content="[CONTEXT]: " + context + " [QUESTION]: " + lastMessage.content)
    messages.append(user_message)
    return messages


def convert_sse(obj: str | dict):
    """Convert the given object (or string) to a Server-Sent Event (SSE) event"""
    # print(obj)
    return "data: {}\n\n".format(json.dumps(obj))
