"""Working with questions in chat messages."""

from telegram import Message
from telegram.ext import (
    CallbackContext,
)
from bot.models import UserData


def extract_private(message: Message, context: CallbackContext) -> str:
    """Extracts a question from a message in a private chat."""
    # allow any messages in a private chat
    question = message.text
    if message.reply_to_message:
        # it's a follow-up question
        question = f"+ {question}"
    return question


def extract_group(message: Message, context: CallbackContext) -> tuple[str, Message]:
    """Extracts a question from a message in a group chat."""
    if (
        message.reply_to_message
        and message.reply_to_message.from_user.username == context.bot.username
    ):
        # treat a reply to the bot as a follow-up question
        question = f"+ {message.text}"
        return question, message

    elif not message.text.startswith(context.bot.name):
        # the message is not a reply to the bot,
        # so ignore it unless it's mentioning the bot
        return "", message

    # the message is mentioning the bot,
    # so remove the mention to get the question
    question = message.text.removeprefix(context.bot.name).strip()

    if message.reply_to_message:
        # the real question is in the original message
        question = (
            f"{question}: {message.reply_to_message.text}"
            if question
            else message.reply_to_message.text
        )
        return question, message.reply_to_message

    return question, message


def prepare(question: str, context: CallbackContext) -> tuple[str, list]:
    """Returns the question along with the previous messages (for follow-up questions)."""
    user = UserData(context.user_data)
    history = []
    if question[0] == "+":
        question = question.strip("+ ")
        history = user.messages.as_list()
    else:
        # user is asking a question 'from scratch',
        # so the bot should forget the previous history
        user.messages.clear()
    return question, history