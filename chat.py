# chat.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Tuple

class KiswahiliChatbot:
    def __init__(self, model_name: str = "gpt2", device: str = None):
        """
        Anzisha chatbot ya Kiswahili.
        :param model_name: Jina la modeli ya Hugging Face au path
        :param device: "cuda" au "cpu"; inagundua kiotomatiki ikiwa ni None
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Inatumia kifaa: {self.device}")

        print(f"Inapakia modeli '{model_name}'...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def chat(self, message: str, history: List[Tuple[str, str]] = None, 
             max_new_tokens: int = 200, temperature: float = 0.7, 
             top_p: float = 0.95) -> str:
        """
        Tengeneza jibu kutoka kwa chatbot.
        :param message: Maandishi ya mtumiaji
        :param history: Historia ya mazungumzo
        :param max_new_tokens: Idadi ya tokeni mpya za kuzalisha
        :param temperature: Kiwango cha ubunifu
        :param top_p: Top-p sampling
        :return: Jibu la chatbot kama string
        """
        # Tengeneza prompt kutoka kwa historia
        prompt = ""
        if history:
            for user_msg, bot_msg in history:
                prompt += f"Mtumiaji: {user_msg}\nBot: {bot_msg}\n"
        prompt += f"Mtumiaji: {message}\nBot:"
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                top_k=50,
                top_p=top_p,
                temperature=temperature,
                repetition_penalty=1.1
            )

        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        response = response.split("Bot:")[-1].strip()
        
        # Toa response clean (ondoa prompt ya awali)
        if prompt in response:
            response = response.replace(prompt, "").strip()
        
        return response

# Create global instance for easy access
bot_instance = KiswahiliChatbot(model_name="gpt2")
