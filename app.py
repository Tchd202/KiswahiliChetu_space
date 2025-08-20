import gradio as gr
from huggingface_hub import InferenceClient
import time

# Initialize Hugging Face Inference Client
client = InferenceClient("HuggingFaceH4/zephyr-7b-beta")

# Language texts
LANGUAGE_TEXTS = {
    "English": {
        "title": "AI Chat - Conversation Service",
        "description": "Chat with your AI assistant",
        "system_default": "You are a friendly assistant who speaks both English and Swahili. Respond in the user's preferred language.",
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
        "examples": [
            ["How are you? Can you tell me about Tanzania?"],
            ["Please suggest some good Swahili books"],
            ["Can you help me translate this to English?"],
            ["Tell me about Swahili culture"]
        ]
    },
    "Kiswahili": {
        "title": "Mazungumzo ya AI - Huduma ya Mazungumzo",
        "description": "Wasiliana na msaidizi wako wa AI",
        "system_default": "Wewe ni msaidizi mwenye urafiki unaozungumza Kiingereza na Kiswahili. Jibu kwa lugha inayopendekezwa na mtumiaji.",
        "placeholder": "Andika ujumbe wako hapa...",
        "thinking": "Inakokotoa...",
        "clear": "Futa Mazungumzo",
        "submit": "Tuma",
        "retry": "Jaribu Tenai",
        "undo": "Rudisha",
        "system_label": "Ujumbe wa Mfumo",
        "max_tokens_label": "Upeo wa Vitokezi kwa Majibu",
        "temperature_label": "Ubunifu (Joto)",
        "top_p_label": "Uchaguzi wa Top-p",
        "language_label": "Lugha ya Mazungumzo",
        "loading": "Inapakia...",
        "error": "Hitilafu imetokea. Tafadhali jaribu tena baadaye.",
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
    system_message: str = "",
    max_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.95,
    language: str = "English"
):
    """
    Handles streaming responses from Hugging Face API and yields output.
    """
    history = history or []
    
    # Use provided system message or default based on language
    if not system_message:
        system_message = get_localized_text(language, "system_default")
    
    # Add language instruction to system message
    if language == "Kiswahili":
        enhanced_system_message = f"{system_message} Jibu kwa Kiswahili kila wakati."
    else:
        enhanced_system_message = f"{system_message} Respond in English always."
    
    messages = [{"role": "system", "content": enhanced_system_message}]

    for user_msg, assistant_msg in history:
        if user_msg:
            messages.append({"role": "user", "content": user_msg})
        if assistant_msg:
            messages.append({"role": "assistant", "content": assistant_msg})

    messages.append({"role": "user", "content": message})

    response = ""
    try:
        for chunk in client.chat_completion(
            messages,
            max_tokens=max_tokens,
            stream=True,
            temperature=temperature,
            top_p=top_p,
        ):
            if chunk.choices and chunk.choices[0].delta.content:
                token = chunk.choices[0].delta.content
                response += token
                yield response
                
    except Exception as e:
        error_message = get_localized_text(language, "error")
        yield f"❌ {error_message}\n\nError: {str(e)}"

def update_ui_language(language):
    """Update UI elements based on selected language"""
    return [
        gr.Markdown.update(value=f"# {get_localized_text(language, 'title')}"),
        gr.Markdown.update(value=get_localized_text(language, "description")),
        gr.Chatbot.update(label=get_localized_text(language, "title")),
        gr.Textbox.update(
            placeholder=get_localized_text(language, "placeholder"),
            label=get_localized_text(language, "placeholder")
        ),
        gr.Button.update(value=get_localized_text(language, "submit")),
        gr.Button.update(value=get_localized_text(language, "clear")),
        gr.Textbox.update(
            value=get_localized_text(language, "system_default"),
            label=get_localized_text(language, "system_label")
        ),
        gr.Slider.update(label=get_localized_text(language, "max_tokens_label")),
        gr.Slider.update(label=get_localized_text(language, "temperature_label")),
        gr.Slider.update(label=get_localized_text(language, "top_p_label")),
        gr.Dropdown.update(label=get_localized_text(language, "language_label")),
        gr.Examples.update(
            examples=get_localized_text(language, "examples"),
            label=f"{get_localized_text(language, 'examples')} ({language})"
        )
    ]

# Create the Gradio interface
with gr.Blocks(
    title="AI Chat - Multilingual",
    theme=gr.themes.Soft(),
    css="""
    .gradio-container {max-width: 900px !important; margin: auto;}
    .chatbot {min-height: 400px; max-height: 500px;}
    """
) as demo:
    
    # Language state
    current_language = gr.State(value="English")
    
    # Header
    title_md = gr.Markdown(value=f"# {get_localized_text('English', 'title')}")
    description_md = gr.Markdown(value=get_localized_text('English', 'description'))
    
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label=get_localized_text('English', 'title'),
                show_copy_button=True,
                bubble_full_width=False
            )
            
            msg = gr.Textbox(
                placeholder=get_localized_text('English', 'placeholder'),
                label=get_localized_text('English', 'placeholder'),
                lines=2
            )
            
            with gr.Row():
                submit_btn = gr.Button(get_localized_text('English', 'submit'), variant="primary")
                clear_btn = gr.Button(get_localized_text('English', 'clear'))
        
        with gr.Column(scale=1):
            system_msg = gr.Textbox(
                value=get_localized_text('English', 'system_default'),
                label=get_localized_text('English', 'system_label'),
                lines=3
            )
            
            max_tokens = gr.Slider(
                minimum=64, maximum=2048, value=512, step=64,
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
    
    # Examples section
    examples = gr.Examples(
        examples=get_localized_text('English', 'examples'),
        inputs=msg,
        label=f"{get_localized_text('English', 'examples')} (English)"
    )
    
    # Event handlers
    submit_event = msg.submit(
        respond,
        inputs=[msg, chatbot, system_msg, max_tokens, temperature, top_p, current_language],
        outputs=chatbot
    ).then(lambda: "", None, msg)
    
    submit_btn.click(
        respond,
        inputs=[msg, chatbot, system_msg, max_tokens, temperature, top_p, current_language],
        outputs=chatbot
    ).then(lambda: "", None, msg)
    
    clear_btn.click(lambda: None, None, chatbot, queue=False)
    
    # Language change handler
    language_dropdown.change(
        update_ui_language,
        inputs=language_dropdown,
        outputs=[
            title_md, description_md, chatbot, msg, submit_btn, clear_btn,
            system_msg, max_tokens, temperature, top_p, language_dropdown, examples
        ]
    ).then(
        lambda x: x,  # Update the current language state
        inputs=language_dropdown,
        outputs=current_language
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        share=False,
        favicon_path=None,
        show_error=True
    )
