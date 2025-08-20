# app.py - SIMPLIFIED VERSION
import gradio as gr
from chat import bot_instance

# Language texts
LANGUAGE_TEXTS = {
    "English": {
        "title": "AI Chat - Local Model",
        "description": "Chat with your local AI assistant",
        "placeholder": "Type your message here...",
        "clear": "Clear Chat",
        "submit": "Submit",
    },
    "Kiswahili": {
        "title": "Mazungumzo ya AI - Modeli ya Ndani",
        "description": "Wasiliana na msaidizi wako wa AI wa ndani",
        "placeholder": "Andika ujumbe wako hapa...",
        "clear": "Futa Mazungumzo",
        "submit": "Tuma",
    }
}

def get_localized_text(language, key):
    return LANGUAGE_TEXTS.get(language, LANGUAGE_TEXTS["English"]).get(key, key)

def respond(message: str, history: list, max_tokens: int = 200, 
           temperature: float = 0.7, top_p: float = 0.95, language: str = "English"):
    try:
        response = bot_instance.chat(
            message=message,
            history=history,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )
        return response
    except Exception as e:
        return f"Error: {str(e)}"

# SIMPLIFIED Gradio interface without language switching complexity
with gr.Blocks(title="Kiswahili Chatbot", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🇹🇿 Kiswahili Chatbot")
    gr.Markdown("Wasiliana na chatbot inayozungumza Kiswahili")
    
    chatbot = gr.Chatbot()
    msg = gr.Textbox(placeholder="Andika ujumbe wako hapa...", lines=2)
    
    with gr.Row():
        submit_btn = gr.Button("Tuma", variant="primary")
        clear_btn = gr.Button("Futa Mazungumzo")
    
    with gr.Accordion("Mipangilio", open=False):
        max_tokens = gr.Slider(64, 512, value=200, label="Upeo wa Vitokezi")
        temperature = gr.Slider(0.1, 2.0, value=0.7, label="Ubunifu (Joto)")
        top_p = gr.Slider(0.1, 1.0, value=0.95, label="Top-p Sampling")
        language = gr.Dropdown(["English", "Kiswahili"], value="Kiswahili", label="Lugha")
    
    # Examples - SIMPLE STATIC VERSION
    gr.Examples(
        examples=[
            ["Habari yako? Unaweza kuniambia kuhusu Tanzania?"],
            ["Tafadhali nipe mapendekezo ya vitabu bora vya Kiswahili"],
            ["Unaweza kunisaidia kutafsiri hii kwa Kiingereza?"],
            ["Eleza kuhusu utamaduni wa Waswahili"]
        ],
        inputs=msg,
        label="Mifano ya Maswali"
    )
    
    # Event handlers - SIMPLIFIED
    def process_message():
        response = respond(msg.value, chatbot.value, max_tokens.value, temperature.value, top_p.value, language.value)
        return response
    
    msg.submit(
        lambda: (process_message(), ""),
        outputs=[chatbot, msg]
    )
    
    submit_btn.click(
        lambda: (process_message(), ""),
        outputs=[chatbot, msg]
    )
    
    clear_btn.click(lambda: None, outputs=chatbot)

if __name__ == "__main__":
    demo.launch(share=False)
