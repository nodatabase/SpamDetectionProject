from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, filters, ContextTypes
from joblib import load
from config import telegram_token
from models.rule_based import is_rule_based_spam
from models.hybrid import weighted_vote

ml_model = load('model_hugging/ml_spam_model2.joblib')

async def message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message:
        text = update.message.text or ""
        chat_id = update.message.chat_id
        user = update.message.from_user

        rule_pred, th = is_rule_based_spam(message=text, threshold=2)

        if th > 2:
            prediction = 'spam'
        else:
            ml_pred = ml_model.predict([text])[0]
            prediction = weighted_vote(ml_pred, rule_pred, th)


        if prediction == 'spam':
            await update.message.delete()

            notify_text = f"Message '{text}' from @{user.username or user.first_name} was removed as spam."
            await context.bot.send_message(chat_id=chat_id, text=notify_text)

            print(f"Deleted spam message from {user.username or user.first_name}: {text}")
        else:
            print(f"Not spam: {text}")


if __name__ == '__main__':
    app = ApplicationBuilder().token(telegram_token).build()
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), message_handler))
    print("App is running")
    app.run_polling()

