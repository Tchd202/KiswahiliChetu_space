import gradio as gr
from chat import bot_instance
import logging
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Language texts
LANGUAGE_TEXTS = {
    "English": {
        "title": "Kiswahili AI Chat",
        "description": "Chat with AI in Swahili and English",
        "placeholder": "Type your message here...",
        "clear": "Clear Chat",
        "submit": "Send",
        "error": "An error occurred",
        "thinking": "Thinking...",
        "examples": "Example Questions"
    },
    "Kiswahili": {
        "title": "Mazungumzo ya AI ya Kiswahili",
        "description": "Wasiliana na AI kwa Kiswahili na Kiingereza",
        "placeholder": "Andika ujumbe wako hapa...",
        "clear": "Futa Mazungumzo",
        "submit": "Tuma",
        "error": "Hitilafu imetokea",
        "thinking": "Inakokotoa...",
        "examples": "Mifano ya Maswali"
    }
}

def get_localized_text(language, key):
    return LANGUAGE_TEXTS.get(language, LANGUAGE_TEXTS["English"]).get(key, key)

def process_message(message, chat_history, max_tokens, temperature, top_p, language):
    """Process message and return updated chat history"""
    try:
        if not message.strip():
            return chat_history, ""
            
        if bot_instance is None:
            error_msg = get_localized_text(language, "error") + ": Chatbot not initialized"
            return chat_history + [(message, error_msg)], ""
        
        # Add thinking message
        thinking_msg = get_localized_text(language, "thinking")
        yield chat_history + [(message, None)], ""
        time.sleep(0.1)  # Small delay for UX
        
        # Get response from chatbot
        response = bot_instance.chat(
            message=message,
            history=chat_history,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )
        
        # Return final response
        yield chat_history + [(message, response)], ""
        
    except Exception as e:
        error_msg = f"{get_localized_text(language, 'error')}: {str(e)}"
        logger.error(f"Error in process_message: {e}")
        yield chat_history + [(message, error_msg)], ""

def clear_chat():
    """Clear chat history"""
    return []

# Create the Gradio interface
with gr.Blocks(
    title="Kiswahili AI Chat",
    theme=gr.themes.Soft(primary_hue="blue", secondary_hue="green"),
    css="""
    .gradio-container { 
        max-width: 1000px !important; 
        margin: auto !important; 
    }
    .chatbot { 
        min-height: 500px; 
        border-radius: 12px; 
        border: 1px solid #e0e0e0;
    }
    .gradio-button {
        border-radius: 8px;
    }
    """
) as demo:
    
    gr.Markdown("""
    # 🇹🇿 Kiswahili AI Chat
    ### Wasiliana na msaidizi wa AI kwa Kiswahili na Kiingereza
    """)
    
    # Chat interface
    with gr.Row(equal_height=False):
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="Mazungumzo",
                show_copy_button=True,
                height=550,
                show_label=False,
                avatar_images=(None, "https://api.dicebear.com/7.x/bottts/svg?seed=swahili"))
            
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Andika ujumbe wako hapa...",
                    show_label=False,
                    lines=2,
                    max_lines=5,
                    container=False,
                    scale=8
                )
                
                with gr.Column(scale=2):
                    submit_btn = gr.Button("📤 Tuma", variant="primary", size="lg")
                    clear_btn = gr.Button("🗑️ Futa", variant="secondary")
        
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Mipangilio")
            
            max_tokens = gr.Slider(
                minimum=50, maximum=300, value=150, step=10,
                label="Urefu wa Majibu",
                info="Idadi ya herufi za jibu"
            )
            
            temperature = gr.Slider(
                minimum=0.1, maximum=1.5, value=0.7, step=0.1,
                label="Kiashiria cha Ubunifu",
                info="Kiwango cha mabadiliko ya majibu"
            )
            
            top_p = gr.Slider(
                minimum=0.1, maximum=1.0, value=0.9, step=0.05,
                label="Uchaguzi wa Top-p",
                info="Kiwango cha uchaguzi wa maneno"
            )
            
            language = gr.Dropdown(
                choices=["Kiswahili", "English"],
                value="Kiswahili",
                label="Lugha ya Interface",
                info="Badilisha lugha ya kiolesura"
            )
            
            gr.Markdown("---")
            gr.Markdown("**📊 Taarifa ya Mfumo:**")
            gr.Markdown(f"**Modeli:** distilgpt2")
            gr.Markdown(f"**Gradio:** 5.43.1")
            gr.Markdown(f"**Transformer:** 4.45.1")
    
    # Examples section
    with gr.Accordion("📚 Mifano ya Maswali", open=True):
        gr.Examples(
            examples=[
                ["Habari yako? Unaweza kuniambia kuhusu Tanzania?"],
                ["Tafadhali nipe mapendekezo ya vitabu bora vya Kiswahili"],
                ["Unaweza kunisaidia kutafsiri hii kwa Kiingereza?"],
                ["Eleza kuhusu utamaduni wa Waswahili"],
                ["Nini maana ya 'Hakuna matata' na 'Asante sana'?"],
                ["Toa mfano wa sentensi kwa Kiswahili"]
            ],
            inputs=msg,
            label="Bonyeza mfano wa swali kujaribu:",
            examples_per_page=3
        )
    
    # Event handlers
    msg.submit(
        fn=process_message,
        inputs=[msg, chatbot, max_tokens, temperature, top_p, language],
        outputs=[chatbot, msg]
    )
    
    submit_btn.click(
        fn=process_message,
        inputs=[msg, chatbot, max_tokens, temperature, top_p, language],
        outputs=[chatbot, msg]
    )
    
    clear_btn.click(
        fn=clear_chat,
        inputs=[],
        outputs=chatbot
    )
    
    # Footer
    gr.Markdown("---")
    gr.Markdown("""
    <div style='text-align: center; color: #666;'>
        <strong>Teknolojia:</strong> Gradio 5.43.1 • Transformers • PyTorch<br>
        <em>Imetengenezwa kwa upendo wa lugha ya Kiswahili</em> 💚
    </div>
    """)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        share=False,
        show_error=True,
        debug=False,
        favicon_path=None
    )
