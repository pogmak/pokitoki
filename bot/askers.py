"""
Asker is an abstraction that sends questions to the AI
and responds to the user with answers provided by the AI.
"""
import base64
import io
import logging
import re
import textwrap

from telegram import Chat, Message
from telegram.constants import MessageLimit, ParseMode
from telegram.ext import CallbackContext

from bot import ai
from bot.config import config
from bot import markdown

logger = logging.getLogger(__name__)


class Asker:
    """Asks AI questions and responds with answers."""

    async def ask(self, question: str, history: list[tuple[str, str]], prompt: str, image_bytearray = None) -> str:
        """Asks AI a question."""
        pass

    async def reply(self, message: Message, context: CallbackContext, answer: str) -> None:
        """Replies with an answer from AI."""
        pass


class TextAsker(Asker):
    """Works with chat completion AI."""

    if config.openai.api_key:
        model = ai.chatgpt.Model(config.openai.model)
    else:
        model = ai.gigachat.Model(config.openai.model)

    async def ask(self, question: str, history: list[tuple[str, str]], prompt: str, image_bytearray=None) -> str:
        """Asks AI a question."""
        return await self.model.ask(question, history, prompt, image_bytearray)

    async def reply(self, message: Message, context: CallbackContext, answer: str) -> None:
        """Replies with an answer from AI."""
        html_answer = markdown.to_html(answer)
        if len(html_answer) <= MessageLimit.MAX_TEXT_LENGTH:
            await message.reply_text(html_answer, parse_mode=ParseMode.HTML)
            return

        doc = io.StringIO(answer)
        caption = (
                textwrap.shorten(answer, width=40, placeholder="...") + " (see attachment for the rest)"
        )
        reply_to_message_id = message.id if message.chat.type != Chat.PRIVATE else None
        await context.bot.send_document(
            chat_id=message.chat_id,
            caption=caption,
            filename=f"{message.id}.md",
            document=doc,
            reply_to_message_id=reply_to_message_id,
        )


class ImagineAsker(Asker):
    """Works with image generation AI."""

    model = ai.dalle.Model()
    size_re = re.compile(r"(256|512|1024)(?:x\1)?\s?(?:px)?")
    sizes = {"256": "256x256", "512": "512x512", "1024": "1024x1024"}
    default_size = "512x512"

    def __init__(self) -> None:
        self.caption = ""

    async def ask(self, question: str, history: list[tuple[str, str]], prompt: str, image_bytearray=None) -> str:
        """Asks AI a question."""
        size = self._extract_size(question)
        self.caption = self._extract_caption(question)
        style = re.search(r"Style:(.*)\s", self.caption)
        try:
            style = style.group(1).strip()
        except:
            logger.info(f"Style didn't found. Use default")
            style = 'DEFAULT'
        return await self.model.imagine(prompt=self.caption, style=style)

    async def reply(self, message: Message, context: CallbackContext, answer: str) -> None:
        """Replies with an answer from AI."""
        photo_bytes_io = io.BytesIO(base64.decodebytes(bytes(answer, "utf-8")))
        await message.reply_photo(photo=photo_bytes_io, caption=self.caption)

    def _extract_size(self, question: str) -> str:
        match = self.size_re.search(question)
        if not match:
            return self.default_size
        width = match.group(1)
        return self.sizes[width]

    def _extract_caption(self, question: str) -> str:
        caption = self.size_re.sub("", question).strip()
        return caption


def create(question: str) -> Asker:
    """Creates a new asker based on the question asked."""
    if question.startswith("/imagine"):
        return ImagineAsker()
    return TextAsker()
