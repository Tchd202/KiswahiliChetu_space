import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import os

# Try to load trained model first, fallback to original
model_path = "./simple-model"  # Your trained model

class SwahiliChatbot:
    def __init__(self):
        print("🚀 Inapakia modeli ya Kiswahili...")
        
        # Try to load trained model first
        if os.path.exists(model_path) and os.path.isdir(model_path):
            try:
                print("✅ Kutumia modeli MPYA iliyofunzwa...")
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                self.model = AutoModelForCausalLM.from_pretrained(model_path)
                self.is_trained_model = True
            except Exception as e:
                print(f"❌ Hitilafu kwa modeli mpya: {e}")
                self.load_fallback_model()
        else:
            print("ℹ️ Modeli mpya haipo. Kutumia modeli ya msingi...")
            self.load_fallback_model()
        
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=-1
        )
        print("✅ Modeli imepakika kikamilifu!")
    
    def load_fallback_model(self):
        """Load fallback model if trained model fails"""
        print("📦 Inapakia modeli ya msingi...")
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
        self.model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
        self.is_trained_model = False
    
    def generate_response(self, message):
        try:
            response = self.pipe(
                message,
                max_length=100,
                temperature=0.7,
                do_sample=True,
                top_k=50,
                num_return_sequences=1
            )[0]['generated_text']
            
            # Clean up response
            if "User:" in response:
                response = response.split("User:")[0].strip()
            
            return response
            
        except Exception as e:
            return f"Samahani, kuna hitilafu: {str(e)}"

# Initialize chatbot
chatbot = SwahiliChatbot()

def chat_interface(message, history):
    response = chatbot.generate_response(message)
    return response

with gr.Blocks(
    title="Kiswahili AI - Modeli Iliyofunzwa",
    theme=gr.themes.Soft(primary_hue="blue")
) as demo:
    gr.Markdown("# 🇹🇿 Kiswahili AI (Modeli Mpya Iliyofunzwa)")
    gr.Markdown("### Wasiliana na modeli ya Kiswahili iliyofunzwa na mafunzo")
    
    if chatbot.is_trained_model:
        gr.Markdown("**🎯 Modeli ya Sasa: Modeli Iliyofunzwa (MPYA!)**")
    else:
        gr.Markdown("**ℹ️ Modeli ya Sasa: Modeli ya Msingi (Fallback)**")
    
    with gr.Row():
        chatbot_ui = gr.Chatbot(height=400, label="Mazungumzo")
        msg = gr.Textbox(
            label="Ujumbe Wako",
            placeholder="Andika hapa kwa Kiswahili...",
            lines=2
        )
    
    with gr.Row():
        submit_btn = gr.Button("📤 Tuma", variant="primary")
        clear_btn = gr.Button("🗑️ Futa Mazungumzo")
    
    # Examples
    gr.Examples(
        examples=[
            ["Habari yako?"],
            ["Jina lako nani?"],
            ["Unaweza kusema Kiswahili?"],
            ["Eleza kuhusu Tanzania"],
            ["Nini maana ya 'Hakuna matata'?"]
        ],
        inputs=msg,
        label="Mifano ya Maswali"
    )
    
    # Footer
    gr.Markdown("---")
    gr.Markdown("**🤖 Modeli:** Kiswahili Fine-tuned | **🚀 Teknolojia:** Gradio + Transformers")
    
    # Event handlers
    def process_message(message, history):
        response = chatbot.generate_response(message)
        return history + [(message, response)]
    
    msg.submit(process_message, [msg, chatbot_ui], [chatbot_ui])
    submit_btn.click(process_message, [msg, chatbot_ui], [chatbot_ui])
    clear_btn.click(lambda: None, None, chatbot_ui)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=False)
