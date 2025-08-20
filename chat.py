import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Tuple

class KiswahiliChatbot:
    def __init__(self, model_name: str = "distilgpt2", device: str = None):
        """
        Very fast chatbot with distilgpt2 model.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Inatumia kifaa: {self.device}")

        print(f"Inapakia modeli '{model_name}'...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        print("Modeli imepakika!")

    def chat(self, message: str, history: List[Tuple[str, str]] = None, 
             max_new_tokens: int = 100, temperature: float = 0.7, 
             top_p: float = 0.9) -> str:
        """
        Faster chat method with optimizations.
        """
        # Simple prompt - faster processing
        prompt = f"User: {message}\nAssistant:"
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=1.1,
                num_return_sequences=1
            )

        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        response = response.split("Assistant:")[-1].strip()
        
        return response

# Create global instance for easy access
bot_instance = KiswahiliChatbot(model_name="distilgpt2")
