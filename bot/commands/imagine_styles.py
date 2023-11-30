"""/start command."""
import requests
from telegram import Update
from telegram.ext import CallbackContext
from telegram.constants import ParseMode

from bot.config import config
from . import constants
from . import help


class ImagineStylesCommand:
    """Answers the `start` command."""

    async def __call__(self, update: Update, context: CallbackContext) -> None:
        styles_json = requests.get("https://cdn.fusionbrain.ai/static/styles/api")
        available_styles = [x['name'] for x in styles_json.json()]
        text = f"Доступные стили для генерации: {available_styles}"
        await update.message.reply_text(
            text, parse_mode=ParseMode.MARKDOWN, disable_web_page_preview=True
        )


