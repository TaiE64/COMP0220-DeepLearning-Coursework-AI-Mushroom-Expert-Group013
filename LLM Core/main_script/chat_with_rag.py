#!/usr/bin/env python3
"""
💬 Mushroom Chatbot with RAG
Real-time knowledge base query during conversation
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import chromadb
from chromadb.utils import embedding_functions
import re

# ================= Configuration =================
# Model configuration
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
LORA_ADAPTER = "./qwen25_mushroom_qlora"  # Your fine-tuned model path

# RAG configuration
CHROMA_DB_PATH = "data/RAG_dataset/chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
USE_RAG = True  # Whether to enable RAG
TOP_K = 3  # Retrieve top 3 documents

# Generation configuration
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7

# Memory optimization (set to True for 8GB VRAM)
USE_4BIT_QUANTIZATION = False  # Set to True to enable 4-bit quantization (good for 8GB VRAM)
# ========================================

class MushroomChatbot:
    def __init__(self, use_rag=USE_RAG):
        """Initialize chatbot"""
        self.use_rag = use_rag
        self.conversation_history = []

        print("🔧 Loading model...")
        self.load_model()

        if self.use_rag:
            print("🔍 Loading RAG knowledge base...")
            self.load_rag()

        print("✅ Initialization complete!\n")

    def load_model(self):
        """Load fine-tuned model"""
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

        # Load base model with optional 4-bit quantization
        if USE_4BIT_QUANTIZATION:
            print("  📦 Using 4-bit quantization (NF4) for memory efficiency...")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            base_model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            base_model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )

        # Load LoRA adapter
        try:
            self.model = PeftModel.from_pretrained(base_model, LORA_ADAPTER)
            print(f"  ✅ Loaded fine-tuned model: {LORA_ADAPTER}")
        except:
            self.model = base_model
            print(f"  ⚠️  Fine-tuned model not found, using base model")

        self.model.eval()
        
        # Display GPU memory usage
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            reserved = torch.cuda.memory_reserved(0) / 1024**3
            print(f"  💾 GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

    def load_rag(self):
        """Load RAG vector database"""
        try:
            client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
            embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=EMBEDDING_MODEL
            )
            self.rag_collection = client.get_collection(
                name="mushroom_knowledge",
                embedding_function=embedding_function
            )
            print(f"  ✅ Loaded knowledge base: {self.rag_collection.count()} documents")
        except Exception as e:
            print(f"  ⚠️  RAG loading failed: {e}")
            self.use_rag = False

    def should_use_rag(self, query: str) -> bool:
        """
        Determine if RAG should be used.
        
        Manual trigger mode: User explicitly requests wiki search with:
        - @search <query>
        - /search <query>
        - search: <query>
        
        This gives users control and reduces unnecessary RAG calls.
        """
        if not self.use_rag:
            return False

        query_lower = query.lower().strip()
        
        # Check for explicit search commands
        search_triggers = ['@search', '/search', 'search:']
        
        return any(query_lower.startswith(trigger) for trigger in search_triggers)
    
    def extract_search_query(self, query: str) -> str:
        """
        Extract the actual query from search command.
        Example: "@search what is amanita" -> "what is amanita"
        """
        query_stripped = query.strip()
        
        # Remove search prefix
        for trigger in ['@search', '/search', 'search:']:
            if query_stripped.lower().startswith(trigger):
                # Remove the trigger and any following whitespace
                return query_stripped[len(trigger):].strip()
        
        return query

    def retrieve_context(self, query: str) -> str:
        """Retrieve relevant context from RAG database"""
        if not self.use_rag:
            return ""

        try:
            results = self.rag_collection.query(
                query_texts=[query],
                n_results=TOP_K
            )

            # Build context
            context_parts = []
            for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0]), 1):
                topic = metadata.get('topic', 'Unknown')
                # Truncate long documents
                doc_preview = doc[:400] if len(doc) > 400 else doc
                context_parts.append(f"[Source {i}: {topic}]\n{doc_preview}")

            context = "\n\n".join(context_parts)
            return context

        except Exception as e:
            print(f"  ⚠️  RAG retrieval failed: {e}")
            return ""

    def generate_response(self, user_input: str) -> str:
        """Generate response"""
        # Determine whether to use RAG
        use_rag_for_this = self.should_use_rag(user_input)

        if use_rag_for_this:
            print("  🔍 [Searching wiki knowledge base...]")
            # Extract the actual query (remove @search, /search, etc.)
            search_query = self.extract_search_query(user_input)
            context = self.retrieve_context(search_query)
            # Use the clean query for the prompt
            actual_query = search_query
        else:
            print("  💬 [Using fine-tuned model directly]")
            context = ""
            actual_query = user_input

        # Build prompt
        if context:
            # With RAG context
            system_message = """You are a knowledgeable mushroom expert assistant. Use the provided context to answer the user's question accurately. If the context doesn't contain the answer, use your general knowledge but mention that."""

            messages = [
                {"role": "system", "content": system_message},
            ]

            # Add conversation history (last 3 rounds)
            for msg in self.conversation_history[-6:]:
                messages.append(msg)

            # Add context and current question
            user_message = f"""Context from knowledge base:
{context}

User question: {actual_query}"""

            messages.append({"role": "user", "content": user_message})

        else:
            # No RAG, direct conversation
            messages = [
                {"role": "system", "content": "You are a helpful and friendly mushroom expert assistant."}
            ]

            # Add conversation history
            for msg in self.conversation_history[-6:]:
                messages.append(msg)

            messages.append({"role": "user", "content": actual_query})

        # Use tokenizer's apply_chat_template
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Generate
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode
        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        return response.strip()

    def chat(self, user_input: str) -> str:
        """Process user input and return response"""
        # Generate response
        response = self.generate_response(user_input)

        # Update conversation history (use clean query without search prefix)
        if self.should_use_rag(user_input):
            clean_query = self.extract_search_query(user_input)
        else:
            clean_query = user_input
            
        self.conversation_history.append({"role": "user", "content": clean_query})
        self.conversation_history.append({"role": "assistant", "content": response})

        return response

    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        print("🗑️  Conversation history cleared")

def main():
    print("=" * 80)
    print("🍄 Mushroom Knowledge Chatbot (with RAG)")
    print("=" * 80)
    print("\n💡 Commands:")
    print("  - Ask questions normally for quick answers from the fine-tuned model")
    print("  - Use '@search <query>' to search the wiki knowledge base")
    print("    Examples:")
    print("      @search what is amanita muscaria")
    print("      /search can I eat this mushroom")
    print("      search: how to identify poisonous mushrooms")
    print("  - 'clear' to clear conversation history")
    print("  - 'quit' or 'exit' to quit")
    print()

    # Initialize chatbot
    bot = MushroomChatbot(use_rag=USE_RAG)

    # Interactive loop
    print("=" * 80)
    print("Let's chat!")
    print("=" * 80 + "\n")

    while True:
        try:
            user_input = input("👤 You: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break

            if user_input.lower() == 'clear':
                bot.clear_history()
                continue

            # Generate response
            print()
            response = bot.chat(user_input)

            print(f"🍄 Assistant: {response}\n")
            print("-" * 80 + "\n")

        except Exception as e:
            print(f"\n❌ Error: {e}\n")

if __name__ == "__main__":
    main()
