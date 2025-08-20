# app.py
import gradio as gr
from chat import bot_instance  # Import the chatbot instance

# Language texts
LANGUAGE_TEXTS = {
    "English": {
        "title": "AI Chat - Local Model",
        "description": "Chat with your local AI assistant",
        "system_default": "You are a friendly assistant who speaks both English and Swahili.",
        "placeholder": "Type your message here...",
        "thinking": "Thinking...",
        "clear": "Clear Chat",
        "submit": "Submit",
        "retry": "Retry",
        "undo": "Undo",
        "system_label": "System Message",
        "max_tokens_label": "Max Tokens per Response",
        "temperature_label": "Creativity (Temperature)",
        "top_p_label": "Top-p Sampling",
        "language_label": "Conversation Language",
        "loading": "Loading...",
        "error": "An error occurred. Please try again later.",
        "examples_label": "Example Questions",
        "examples": [
            ["How are you? Can you tell me about Tanzania?"],
            ["Please suggest some good Swahili books"],
            ["Can you help me translate this to English?"],
            ["Tell me about Swahili culture"]
        ]
    },
    "Kiswahili": {
        "title": "Mazungumzo ya AI - Modeli ya Ndani",
        "description": "Wasiliana na msaidizi wako wa AI wa ndani",
        "system_default": "Wewe ni msaidizi mwenye urafiki unaozungumza Kiingereza na Kiswahili.",
        "placeholder": "Andika ujumbe wako hapa...",
        "thinking": "Inakokotoa...",
        "clear": "Futa Mazungumzo",
        "submit": "Tuma",
        "retry": "Jaribu Tena",
        "undo": "Rudisha",
        "system_label": "Ujumbe wa Mfumo",
        "max_tokens_label": "Upeo wa Vitokezi kwa Majibu",
        "temperature_label": "Ubunifu (Joto)",
        "top_p_label": "Uchaguzi wa Top-p",
        "language_label": "Lugha ya Mazungumzo",
        "loading": "Inapakia...",
        "error": "Hitilafu imetokea. Tafadhali jaribu tena baadaye.",
        "examples_label": "Mifano ya Maswali",
        "examples": [
            ["Habari yako? Unaweza kuniambia kuhusu Tanzania?"],
            ["Tafadhali nipe mapendekezo ya vitabu bora vya Kiswahili"],
            ["Unaweza kunisaidia kutafsiri hii kwa Kiingereza?"],
            ["Eleza kuhusu utamaduni wa Waswahili"]
        ]
    }
}

def get_localized_text(language, key):
    """Get localized text based on selected language"""
    return LANGUAGE_TEXTS.get(language, LANGUAGE_TEXTS["English"]).get(key, key)

def respond(
    message: str,
    history: list[tuple[str, str]] = None,
    max_tokens: int = 200,
    temperature: float = 0.7,
    top_p: float = 0.95,
    language: str = "English"
):
    """
    Handles responses using the local chatbot model.
    """
    try:
        # Use the local chatbot instance
        response = bot_instance.chat(
            message=message,
            history=history,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )
        return response
    except Exception as e:
        error_message = get_localized_text(language, "error")
        return f"❌ {error_message}\n\nError: {str(e)}"

def update_ui_language(language):
    """Update UI elements based on selected language"""
    return [
        gr.Markdown.update(value=f"# {get_localized_text(language, 'title')}"),
        gr.Markdown.update(value=get_localized_text(language, "description")),
        gr.Chatbot.update(label=get_localized_text(language, "title")),
        gr.Textbox.update(placeholder=get_localized_text(language, "placeholder")),
        gr.Button.update(value=get_localized_text(language, "submit")),
        gr.Button.update(value=get_localized_text(language, "clear")),
        gr.Slider.update(label=get_localized_text(language, "max_tokens_label")),
        gr.Slider.update(label=get_localized_text(language, "temperature_label")),
        gr.Slider.update(label=get_localized_text(language, "top_p_label")),
        gr.Dropdown.update(label=get_localized_text(language, "language_label"))
    ]

# Create the Gradio interface
with gr.Blocks(
    title="Kiswahili Chatbot - Local",
    theme=gr.themes.Soft(),
    css="""
    .gradio-container {max-width: 900px !important; margin: auto;}
    .chatbot {min-height: 400px; max-height: 500px;}
    """
) as demo:
    
    current_language = gr.State(value="English")
    
    # Header
    title_md = gr.Markdown(value=f"# {get_localized_text('English', 'title')}")
    description_md = gr.Markdown(value=get_localized_text('English', 'description'))
    
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label=get_localized_text('English', 'title'),
                show_copy_button=True
            )
            
            msg = gr.Textbox(
                placeholder=get_localized_text('English', 'placeholder'),
                label="Message",
                lines=2
            )
            
            with gr.Row():
                submit_btn = gr.Button(get_localized_text('English', 'submit'), variant="primary")
                clear_btn = gr.Button(get_localized_text('English', 'clear'))
        
        with gr.Column(scale=1):
            max_tokens = gr.Slider(
                minimum=64, maximum=512, value=200, step=32,
                label=get_localized_text('English', 'max_tokens_label')
            )
            
            temperature = gr.Slider(
                minimum=0.1, maximum=2.0, value=0.7, step=0.1,
                label=get_localized_text('English', 'temperature_label')
            )
            
            top_p = gr.Slider(
                minimum=0.1, maximum=1.0, value=0.95, step=0.05,
                label=get_localized_text('English', 'top_p_label')
            )
            
            language_dropdown = gr.Dropdown(
                choices=["English", "Kiswahili"],
                value="English",
                label=get_localized_text('English', 'language_label')
            )
    
    # Examples sections
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Example Questions (English)")
            gr.Examples(
                examples=LANGUAGE_TEXTS["English"]["examples"],
                inputs=msg,
                label=""
            )
        with gr.Column():
            gr.Markdown("### Mifano ya Maswali (Kiswahili)")
            gr.Examples(
                examples=LANGUAGE_TEXTS["Kiswahili"]["examples"],
                inputs=msg,
                label=""
            )
    
    # Event handlers
    def clear_and_reset():
        return None, ""
    
    msg.submit(
        respond,
        inputs=[msg, chatbot, max_tokens, temperature, top_p, current_language],
        outputs=chatbot
    ).then(clear_and_reset, outputs=[chatbot, msg])
    
    submit_btn.click(
        respond,
        inputs=[msg, chatbot, max_tokens, temperature, top_p, current_language],
        outputs=chatbot
    ).then(clear_and_reset, outputs=[chatbot, msg])
    
    clear_btn.click(lambda: None, None, chatbot)
    
    # Language change handler
    language_dropdown.change(
        update_ui_language,
        inputs=language_dropdown,
        outputs=[
            title_md, description_md, chatbot, msg, submit_btn, clear_btn,
            max_tokens, temperature, top_p, language_dropdown
        ]
    ).then(lambda x: x, inputs=language_dropdown, outputs=current_language)

if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0")
