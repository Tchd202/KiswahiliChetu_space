# chat.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        self.chat_history = []

    def chat(self, user_input: str, max_new_tokens: int = 200):
        """
        Tengeneza jibu kutoka kwa chatbot.
        :param user_input: Maandishi ya mtumiaji
        :param max_new_tokens: Idadi ya tokeni mpya za kuzalisha
        :return: Jibu la chatbot kama string
        """
        self.chat_history.append(f"Mtumiaji: {user_input}")
        prompt = "\n".join(self.chat_history) + "\nBot:"

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.7
            )

        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        response = response.split("Bot:")[-1].strip()

        self.chat_history.append(f"Bot: {response}")
        return response

    def reset_history(self):
        """Futa historia ya mazungumzo."""
        self.chat_history = []

if __name__ == "__main__":
    bot = KiswahiliChatbot(model_name="gpt2")  # Badilisha na modeli yako ya Kiswahili
    print("Chatbot ya Kiswahili iko tayari! Andika 'toka' kuondoka.\n")

    while True:
        user_input = input("Wewe: ")
        if user_input.lower() in ["toka", "ondoka"]:
            print("Chatbot inafunga...")
            break
        response = bot.chat(user_input)
        print(f"Bot: {response}\n")

