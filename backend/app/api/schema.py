from typing import List

from llama_index.core.llms import MessageRole
from pydantic import BaseModel


class Message(BaseModel):
    role: MessageRole
    content: str


class ChatMessages(BaseModel):
    messages: List[Message]
