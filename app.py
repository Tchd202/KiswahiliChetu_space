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

def process_message(message, history, max_tokens, temperature, top_p, language):
    try:
        response = bot_instance.chat(
            message=message,
            history=history,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )
        return history + [(message, response)], ""
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        return history + [(message, error_msg)], ""

# SIMPLIFIED Gradio interface
with gr.Blocks(title="Kiswahili Chatbot", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🇹🇿 Kiswahili Chatbot")
    gr.Markdown("Wasiliana na chatbot inayozungumza Kiswahili")
    
    chatbot = gr.Chatbot()
    msg = gr.Textbox(placeholder="Andika ujumbe wako hapa...", lines=2)
    
    with gr.Row():
        submit_btn = gr.Button("Tuma", variant="primary")
        clear_btn = gr.Button("Futa Mazungumzo")
    
    with gr.Accordion("Mipangilio", open=False):
        max_tokens = gr.Slider(64, 512, value=100, label="Upeo wa Vitokezi")
        temperature = gr.Slider(0.1, 2.0, value=0.7, label="Ubunifu (Joto)")
        top_p = gr.Slider(0.1, 1.0, value=0.9, label="Top-p Sampling")
        language = gr.Dropdown(["English", "Kiswahili"], value="Kiswahili", label="Lugha")
    
    # Examples
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
    
    # Event handlers
    msg.submit(
        process_message,
        inputs=[msg, chatbot, max_tokens, temperature, top_p, language],
        outputs=[chatbot, msg]
    )
    
    submit_btn.click(
        process_message,
        inputs=[msg, chatbot, max_tokens, temperature, top_p, language],
        outputs=[chatbot, msg]
    )
    
    clear_btn.click(lambda: None, outputs=chatbot)

if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0")
