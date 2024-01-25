"""Gigachat from sber."""

import logging
import tiktoken
from langchain.chat_models.gigachat import GigaChat
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage, BaseMessage

from bot.config import config

logger = logging.getLogger(__name__)

credentials = config.gigachat.get('api_key')
gigachat = GigaChat(credentials=credentials, verify_ssl_certs=False)

encoding = tiktoken.get_encoding("cl100k_base")


class Model:
    """OpenAI API wrapper."""

    def __init__(self, name: str) -> None:
        """Creates a wrapper for a given OpenAI large language model."""
        self.name = name

    async def ask(self, question: str, history: list[tuple[str, str]], prompt: str) -> str:
        """Asks the language model a question and returns an answer."""
        # maximum number of input tokens
        n_input = _calc_n_input(n_output=config.openai.params["max_tokens"])
        messages = self._generate_messages(question, history, prompt)
        messages = shorten(messages, length=n_input)
        logger.info(f"Sending message chain: {messages}")
        resp = gigachat(messages)
        logger.debug(f"Response: {resp}")
        answer = resp.content
        return answer

    def _generate_messages(self, question: str, history: list[tuple[str, str]], prompt: str) -> list[BaseMessage]:
        """Builds message history to provide context for the language model."""
        messages = [SystemMessage(content=prompt),]
        for prev_question, prev_answer in history:
            messages.append(HumanMessage(content=prev_question))
            messages.append(AIMessage(content=prev_answer))
        messages.append(HumanMessage(content=question))
        return messages


def shorten(messages: list[BaseMessage], length: int) -> list[BaseMessage]:
    """
    Truncates messages so that the total number or tokens
    does not exceed the specified length.
    """
    lengths = [gigachat.get_num_tokens(m.content) for m in messages]
    total_len = sum(lengths)
    if total_len <= length:
        return messages

    # exclude older messages to fit into the desired length
    # can't exclude the prompt though
    prompt_msg, messages = messages[0], messages[1:]
    prompt_len, lengths = lengths[0], lengths[1:]
    while len(messages) > 1 and total_len > length:
        messages = messages[1:]
        first_len, lengths = lengths[0], lengths[1:]
        total_len -= first_len
    messages = [prompt_msg] + messages
    if total_len <= length:
        return messages

    return messages


def _calc_n_input(n_output: int) -> int:
    """
    Calculates the maximum number of input tokens
    according to the model and the maximum number of output tokens.
    """
    n_total = 4096  # max 4096 tokens total by default
    return n_total - n_output
