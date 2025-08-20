import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Tuple, Optional
import logging
from functools import lru_cache

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KiswahiliChatbot:
    def __init__(self, model_name: str = "distilgpt2", device: str = None):
        """
        Fast chatbot with distilgpt2 model (334MB) - Optimized for Gradio 5.x
        """
        try:
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Inatumia kifaa: {self.device}")

            logger.info(f"Inapakia modeli '{model_name}'...")
            
            # Load with optimizations
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir="./model_cache"
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                cache_dir="./model_cache",
                low_cpu_mem_usage=True
            )
            
            self.model.to(self.device)
            
            if self.device == "cuda":
                self.model = self.model.half()  # Use half precision for faster inference
            
            self.model.eval()
            logger.info("✅ Modeli imepakika kikamilifu!")
            
        except Exception as e:
            logger.error(f"❌ Hitilafu wakati wa kupakia modeli: {e}")
            raise

    @torch.no_grad()
    def chat(self, message: str, history: Optional[List[Tuple[str, str]]] = None, 
             max_new_tokens: int = 150, temperature: float = 0.7, 
             top_p: float = 0.9) -> str:
        """
        Optimized chat method for faster responses.
        """
        try:
            if not message.strip():
                return "Tafadhali andika ujumbe..."
            
            # Build prompt from history
            prompt = self._build_prompt(message, history)
            
            # Tokenize with truncation
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=1024
            ).to(self.device)
            
            # Generate response
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                temperature=max(0.1, min(temperature, 2.0)),
                top_p=max(0.1, min(top_p, 1.0)),
                top_k=50,
                repetition_penalty=1.2,
                num_return_sequences=1,
                early_stopping=True
            )

            # Decode and clean response
            response = self.tokenizer.decode(
                output_ids[0], 
                skip_special_tokens=True
            )
            
            # Extract only the assistant's response
            if "Assistant:" in response:
                response = response.split("Assistant:")[-1].strip()
            else:
                # If no assistant marker, take the last part
                response = response.replace(prompt, "").strip()
            
            # Clean up response
            response = self._clean_response(response)
            
            return response if response else "Samahani, sijaweza kutengeneza jibu. Tafadhali jaribu tena."
            
        except Exception as e:
            logger.error(f"❌ Hitilafu wakati wa kukokotoa jibu: {e}")
            return f"Samahani, kuna hitilafu ya kiufundi: {str(e)}"

    def _build_prompt(self, message: str, history: Optional[List[Tuple[str, str]]]) -> str:
        """Build conversation prompt from history"""
        prompt = ""
        
        if history:
            for user_msg, bot_msg in history[-6:]:  # Keep last 6 exchanges
                prompt += f"User: {user_msg}\nAssistant: {bot_msg}\n"
        
        prompt += f"User: {message}\nAssistant:"
        return prompt

    def _clean_response(self, response: str) -> str:
        """Clean up the generated response"""
        # Remove any remaining prompt fragments
        response = response.split("User:")[0].strip()
        
        # Remove multiple newlines
        response = ' '.join(response.split())
        
        # Ensure proper sentence structure
        if response and not response.endswith(('.', '!', '?')):
            response += '.'
            
        return response

# Create global instance with error handling
try:
    bot_instance = KiswahiliChatbot(model_name="distilgpt2")
    logger.info("🚀 Chatbot imeanzishwa kikamilifu!")
except Exception as e:
    logger.error(f"💥 Imeshindwa kuanzisha chatbot: {e}")
    bot_instance = None
