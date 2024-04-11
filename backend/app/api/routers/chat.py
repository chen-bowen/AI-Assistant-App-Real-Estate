from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from llama_index.core.indices.vector_store import VectorStoreIndex
from llama_index.core.llms import MessageRole

from app.api.api_utils import build_context_from_index, build_message_sequence
from app.api.schema import ChatMessages
from app.clients.llm_client import LLMClient, get_llm
from app.engine.index import get_index

chat_router = APIRouter()


@chat_router.post("")
async def chat(
    request: Request,
    data: ChatMessages,
    index: VectorStoreIndex = Depends(get_index),
    llm: LLMClient = Depends(get_llm),
) -> StreamingResponse:
    """
    Handle the chat endpoint.

    This function receives a POST request with chat data and processes it using a chat engine.
    It performs the following steps:
    1. Check if any messages are provided. If not, raise an HTTPException with status code 400.
    2. Get the last message from the provided data. If the last message is not from the user, raise an HTTPException with status code 400.
    3. Convert the messages from the request to the ChatMessage type.
    4. Query the chat engine using the last message content and the converted messages.
    5. Stream the response to the client.

    Parameters:
    - request: The incoming fastAPI request object.
    - data: The chat data received in the request.
    - chat_engine: The chat engine to use for processing the chat data.

    Returns:
    - A StreamingResponse object that streams the response to the client.

    Raises:
    - HTTPException with status code 400 if no messages are provided or if the last message is not from the user.
    """
    # check preconditions and get last message
    if len(data.messages) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No messages provided",
        )
    # should be a user message
    lastMessage = data.messages.pop()

    # assert last message is from user
    if lastMessage.role != MessageRole.USER:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Last message must be from user",
        )

    # get context from index and last message
    context = build_context_from_index(index, lastMessage)
    # build message sequence
    messages = build_message_sequence(data, context, lastMessage)

    # stream chat
    response = llm.stream_chat(messages)

    # stream response
    async def event_generator():
        for token in response:
            # If client closes connection, stop sending events
            if await request.is_disconnected():
                break
            yield token.delta

    return StreamingResponse(event_generator(), media_type="text/event-stream")
    # return response.response
