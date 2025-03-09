import os
import sqlite3
import asyncio
import requests
import logging
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    filters,
    ContextTypes,
)
from telegram.constants import ChatAction, ParseMode
from telegram.error import NetworkError, BadRequest
from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai.types.generation_types import StopCandidateException, BlockedPromptException
from google.generativeai.types.safety_types import HarmCategory, HarmBlockThreshold
import PIL.Image as load_image
from io import BytesIO
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("bot.log"),  # Log to file
        logging.StreamHandler()          # Log to console
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configuration
BOT_TOKEN = os.getenv("BOT_TOKEN")
API_KEYS = {
    "openai": os.getenv("OPENAI_API_KEY"),
    "gemini": os.getenv("GOOGLE_API_KEY"),
    "deepseek": os.getenv("DEEPSEEK_API_KEY")
}
LLM_OPTIONS = {
    "OpenAI": ["o1-mini","o1","o3-mini","gpt-4o", "gpt-4o-mini","gpt-3.5-turbo"],
    "Gemini": ["gemini-2.0-flash","gemini-2.0-flash-lite", "gemini-1.5-pro"],
    "DeepSeek": ["deepseek-chat"]
}
_AUTHORIZED_USERS = [i.strip() for i in os.getenv("AUTHORIZED_USERS", "").split(",") if i.strip()]

# Gemini safety settings
SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
}

# Initialize Gemini models
genai.configure(api_key=API_KEYS["gemini"])
GEMINI_MODELS = {
    "gemini-2.0-flash": genai.GenerativeModel("gemini-2.0-flash", safety_settings=SAFETY_SETTINGS),
    "gemini-1.5-pro": genai.GenerativeModel("gemini-1.5-pro", safety_settings=SAFETY_SETTINGS),
}

# --- Database Setup (SQLite) ---
def init_db():
    conn = sqlite3.connect("personal_ai.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users 
                 (telegram_id TEXT PRIMARY KEY, preferred_llm TEXT, preferred_model TEXT)''')
    conn.commit()
    conn.close()
    logger.info("Database initialized")

def set_user_preference(telegram_id: str, llm: str, model: str):
    conn = sqlite3.connect("personal_ai.db")
    c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO users (telegram_id, preferred_llm, preferred_model) VALUES (?, ?, ?)", 
              (telegram_id, llm, model))
    conn.commit()
    conn.close()
    logger.info(f"Set preference : {llm} ({model})")

def get_user_preference(telegram_id: str) -> tuple:
    conn = sqlite3.connect("personal_ai.db")
    c = conn.cursor()
    c.execute("SELECT preferred_llm, preferred_model FROM users WHERE telegram_id = ?", (telegram_id,))
    result = c.fetchone()
    conn.close()
    if result:
        logger.info(f"Retrieved preference for : {result[0]} ({result[1]})")
        return result
    logger.info(f"No preference found for , using default")
    return ("Gemini", "gemini-2.0-flash")

# --- Filters ---
class AuthorizedUserFilter(filters.UpdateFilter):
    def filter(self, update: Update):
        if not _AUTHORIZED_USERS:
            return True
        authorized = (
            update.message.from_user.username in _AUTHORIZED_USERS
            or str(update.message.from_user.id) in _AUTHORIZED_USERS
        )
        logger.info(f"Authorization check for {update.message.from_user.username}: {'Authorized' if authorized else 'Unauthorized'}")
        return authorized

AuthFilter = AuthorizedUserFilter()
MessageFilter = AuthFilter & ~filters.COMMAND & filters.TEXT
PhotoFilter = AuthFilter & ~filters.COMMAND & filters.PHOTO

# --- LLM Call Functions ---
def call_openai(prompt: str, model: str, is_image: bool = False, image=None) -> str:
    logger.info(f"Calling OpenAI with model {model}")
    headers = {"Authorization": f"Bearer {API_KEYS['openai']}", "Content-Type": "application/json"}
    data = {"model": model, "max_tokens": 500}
    if is_image and image:
        return "OpenAI image processing not implemented yet."
    else:
        data["messages"] = [{"role": "user", "content": prompt}]
    response = requests.post("https://api.openai.com/v1/chat/completions", json=data, headers=headers)
    return response.json()["choices"][0]["message"]["content"]

async def call_gemini(prompt: str, model: str, is_image: bool = False, image=None):
    logger.info(f"Calling Gemini with model {model}")
    gemini_model = GEMINI_MODELS.get(model, GEMINI_MODELS["gemini-2.0-flash"])
    if is_image and image:
        return gemini_model.generate_content_async([prompt, image], stream=True)
    else:
        chat = gemini_model.start_chat()
        return await chat.send_message_async(prompt, stream=True)

def call_deepseek(prompt: str, model: str, is_image: bool = False, image=None) -> str:
    logger.info(f"Calling DeepSeek with model {model}")
    headers = {"Authorization": f"Bearer {API_KEYS['deepseek']}", "Content-Type": "application/json"}
    data = {"prompt": prompt, "model": model}
    if is_image and image:
        return "DeepSeek image processing not implemented yet."
    response = requests.post("https://api.deepseek.com/v1/chat", json=data, headers=headers)
    return response.json()["response"]

async def call_llm(prompt: str, llm: str, model: str, is_image: bool = False, image=None):
    try:
        if llm == "OpenAI":
            return call_openai(prompt, model, is_image, image), False
        elif llm == "Gemini":
            return await call_gemini(prompt, model, is_image, image), True
        elif llm == "DeepSeek":
            return call_deepseek(prompt, model, is_image, image), False
        return "Unsupported LLM", False
    except Exception as e:
        logger.error(f"LLM call failed: {str(e)}")
        return f"Error: {str(e)}", False

# --- HTML Formatting ---
def escape_html(text: str) -> str:
    return text.replace("&", "&").replace("<", "<").replace(">", ">")

def apply_hand_points(text: str) -> str:
    return re.sub(r"(?<=\n)\*\s(?!\*)|^\*\s(?!\*)", "👉 ", text)

def apply_bold(text: str) -> str:
    return re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)

def apply_italic(text: str) -> str:
    return re.sub(r"(?<!\*)\*(?!\*)(?!\*\*)(.*?)(?<!\*)\*(?!\*)", r"<i>\1</i>", text)

def apply_code(text: str) -> str:
    return re.sub(r"```([\w]*?)\n([\s\S]*?)```", r"<pre lang='\1'>\2</pre>", text, flags=re.DOTALL)

def apply_monospace(text: str) -> str:
    return re.sub(r"(?<!`)`(?!`)(.*?)(?<!`)`(?!`)", r"<code>\1</code>", text)

def apply_link(text: str) -> str:
    return re.sub(r"\[(.*?)\]\((.*?)\)", r'<a href="\2">\1</a>', text)

def apply_underline(text: str) -> str:
    return re.sub(r"__(.*?)__", r"<u>\1</u>", text)

def apply_strikethrough(text: str) -> str:
    return re.sub(r"~~(.*?)~~", r"<s>\1</s>", text)

def apply_header(text: str) -> str:
    return re.sub(r"^(#{1,6})\s+(.*)", r"<b><u>\2</u></b>", text, flags=re.DOTALL)

def apply_exclude_code(text: str) -> str:
    lines = text.split("\n")
    in_code_block = False
    for i, line in enumerate(lines):
        if line.startswith("```"):
            in_code_block = not in_code_block
        if not in_code_block:
            formatted_line = apply_header(line)
            formatted_line = apply_link(formatted_line)
            formatted_line = apply_bold(formatted_line)
            formatted_line = apply_italic(formatted_line)
            formatted_line = apply_underline(formatted_line)
            formatted_line = apply_strikethrough(formatted_line)
            formatted_line = apply_monospace(formatted_line)
            formatted_line = apply_hand_points(formatted_line)
            lines[i] = formatted_line
    return "\n".join(lines)

def format_message(text: str) -> str:
    formatted_text = escape_html(text)
    formatted_text = apply_exclude_code(formatted_text)
    formatted_text = apply_code(formatted_text)
    return formatted_text

# --- Handlers ---
async def start(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    logger.info(f"User issued /start")
    user = update.effective_user
    welcome_message = (
    f"🌟 <b>Hi, {user.mention_html()}!</b> 🌟\n\n"
    "Welcome to MXC AI Bot! 🚀\n\n"
    "🤖 I'm your personal AI assistant, ready to chat, brainstorm, and assist you!\n\n"
    "✨ Choose your AI with <b>/setllm</b>\n"
    f"🧠 Available LLMs\n"
    + "\n".join([f"🔹 <b>{llm}:</b> {', '.join(models)}" for llm, models in LLM_OPTIONS.items()]) +
    "\n\n"
    "🧠 Pick the best model for your needs!\n"
    "🔄 Use <b>/new</b> anytime to refresh our conversation.\n\n"
    "💬 Send me a message or an 📸 image, and let’s get started!"
)
    await update.message.reply_text(welcome_message, parse_mode=ParseMode.HTML)

async def help_command(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    logger.info(f"User issued /help")
    help_text = """
<b>🎯 Commands:</b>
/start - Kick things off with a warm welcome! 🌟
/help - See this handy guide 📖
/setllm - Choose your AI models 🎛️
/new - Start a fresh chat session 🔄

📝 Send a message or 📸 an image to chat with your selected AI!
"""
    await update.message.reply_text(help_text, parse_mode=ParseMode.HTML)

async def set_llm(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.info(f"User issued /setllm")
    keyboard = [
        [InlineKeyboardButton(f"{llm}", callback_data=f"llm_{llm}") for llm in LLM_OPTIONS.keys()]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text("🤖 Choose your LLM:", reply_markup=reply_markup)

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    data = query.data
    telegram_id = str(query.from_user.id)

    if data.startswith("llm_"):
        llm = data.split("_")[1]
        context.user_data["selected_llm"] = llm
        keyboard = [
            [InlineKeyboardButton(f"✨{model}✨", callback_data=f"model_{model}")] for model in LLM_OPTIONS[llm]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text(f"✅ Selected LLM: {llm}.\n  Now pick a model 🧠", reply_markup=reply_markup)
        logger.info(f"User selected LLM: {llm} ")
    elif data.startswith("model_"):
        model = data.split("_")[1]
        llm = context.user_data.get("selected_llm")
        if llm and model in LLM_OPTIONS[llm]:
            set_user_preference(telegram_id, llm, model)
            await query.edit_message_text(f"🎉 Set LLM to {llm} ({model}) ⚙️ ")
        else:
            await query.edit_message_text("❌ Invalid selection. Use /setllm to try again.")
            logger.warning(f"User made invalid model selection: {model} for {llm}")

async def newchat_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.info(f"User {update.effective_user.id} issued /new")
    init_msg = await update.message.reply_text("🔄 Starting new chat session...")
    await init_msg.edit_text("✅ New chat session started!")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    telegram_id = str(update.message.from_user.id)
    llm, model = get_user_preference(telegram_id)
    logger.info(f"Handling message from {telegram_id} with {llm} ({model}): {update.message.text}")
    init_msg = await update.message.reply_text(f"[{model}] 🚀 Generating...", reply_to_message_id=update.message.message_id)
    await update.message.chat.send_action(ChatAction.TYPING)

    text = update.message.text
    response, is_streaming = await call_llm(text, llm, model)
    if is_streaming:
        full_message = ""
        try:
            async for chunk in response:
                if chunk.text:
                    full_message += chunk.text
                    formatted_message = format_message(full_message)
                    await init_msg.edit_text(formatted_message, parse_mode=ParseMode.HTML, disable_web_page_preview=True)
                    await asyncio.sleep(0.1)
            logger.info(f"Streamed response completed for {telegram_id}")
        except (StopCandidateException, BlockedPromptException) as e:
            await init_msg.edit_text(f"⚠️ Error: {str(e)}")
            logger.error(f"Streaming error for {telegram_id}: {str(e)}")
        except Exception as e:
            await init_msg.edit_text(f"❌ Unexpected error: {str(e)}")
            logger.error(f"Unexpected error for {telegram_id}: {str(e)}")
    else:
        try:
            formatted_message = format_message(response)
            await init_msg.edit_text(formatted_message, parse_mode=ParseMode.HTML, disable_web_page_preview=True)
            logger.info(f"Non-streamed response sent for {telegram_id}")
        except Exception as e:
            await init_msg.edit_text(f"❌ Error: {str(e)}")
            logger.error(f"Error for {telegram_id}: {str(e)}")

async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    telegram_id = str(update.message.from_user.id)
    llm, model = get_user_preference(telegram_id)
    logger.info(f"Handling image from {telegram_id} with {llm} ({model})")
    init_msg = await update.message.reply_text(f"[{model}] 🚀 Generating...", reply_to_message_id=update.message.message_id)

    images = update.message.photo
    unique_images = {}
    for img in images:
        file_id = img.file_id[:-7]
        if file_id not in unique_images or img.file_size > unique_images[file_id].file_size:
            unique_images[file_id] = img
    file = await list(unique_images.values())[0].get_file()
    image = load_image.open(BytesIO(await file.download_as_bytearray()))
    prompt = update.message.caption or "Analyze this image and generate a response"

    response, is_streaming = await call_llm(prompt, llm, model, is_image=True, image=image)
    if is_streaming:
        full_message = ""
        try:
            async for chunk in response:
                if chunk.text:
                    full_message += chunk.text
                    formatted_message = format_message(full_message)
                    await init_msg.edit_text(formatted_message, parse_mode=ParseMode.HTML, disable_web_page_preview=True)
                    await asyncio.sleep(0.1)
            logger.info(f"Streamed image response completed for {telegram_id}")
        except Exception as e:
            await init_msg.edit_text(f"❌ Error processing image: {str(e)}")
            logger.error(f"Image processing error for {telegram_id}: {str(e)}")
    else:
        try:
            formatted_message = format_message(response)
            await init_msg.edit_text(formatted_message, parse_mode=ParseMode.HTML, disable_web_page_preview=True)
            logger.info(f"Non-streamed image response sent for {telegram_id}")
        except Exception as e:
            await init_msg.edit_text(f"❌ Error processing image: {str(e)}")
            logger.error(f"Image error for {telegram_id}: {str(e)}")

# --- Start Bot ---
def start_bot():
    logger.info("Starting bot...")
    init_db()
    application = Application.builder().token(BOT_TOKEN).build()
    application.add_handler(CommandHandler("start", start, filters=AuthFilter))
    application.add_handler(CommandHandler("help", help_command, filters=AuthFilter))
    application.add_handler(CommandHandler("setllm", set_llm, filters=AuthFilter))
    application.add_handler(CommandHandler("new", newchat_command, filters=AuthFilter))
    application.add_handler(MessageHandler(MessageFilter, handle_message))
    application.add_handler(MessageHandler(PhotoFilter, handle_image))
    application.add_handler(CallbackQueryHandler(button_handler))
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    start_bot()
