#!/usr/bin/env python3
"""
🍄 Integrated Mushroom Assistant: Image Detection + Chat System
Combines a Two-Stage Pipeline (YOLO + ViT) with a RAG Chatbot

Two-Stage Pipeline:
- Stage 1: YOLO detects mushroom locations
- Stage 2: ViT identifies mushroom species and toxicity (16 species)
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from peft import PeftModel
import chromadb
from chromadb.utils import embedding_functions
import re
import sys
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
from torchvision import transforms
from PIL import Image
import time
import warnings
import logging
from typing import List, Dict, Optional
from collections import deque

# Suppress all warnings during initialization and runtime
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*logits.*model output.*FP32.*")
warnings.filterwarnings("ignore", message=".*Starting from v4.46.*")
warnings.filterwarnings("ignore", message=".*TypedStorage.*")
warnings.filterwarnings("ignore", message=".*clean_up_tokenization_spaces.*")
warnings.filterwarnings("ignore", message=".*return_all_scores.*")
warnings.filterwarnings("ignore", message=".*deprecated.*")

# Suppress transformers library warnings
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)

# ================= Configuration =================
# Model configuration
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
LORA_ADAPTER = "../qwen25_mushroom_qlora"

# RAG configuration
CHROMA_DB_PATH = "../data/RAG_dataset/chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
USE_RAG = True
TOP_K = 3

# Image Pipeline configuration (ViT version)
DETECTOR_PATH = "../../ImageModel/FT_YOLO_Detection/yolo_detector/weights/best.pt"
CLASSIFIER_PATH = "../../ImageModel/FT_ViT/vit_antioverfit/best_model.pth"
MAPPING_FILE = "../../Dataset/mushroom_species_dataset/species_toxicity_mapping.json"

# Generation configuration
MAX_NEW_TOKENS = 512  # Increase to 512 to support detailed visual feature descriptions
TEMPERATURE = 0.7

# 8GB VRAM optimization configuration
USE_4BIT = True

# Sentiment Analysis configuration
ENABLE_SENTIMENT_ANALYSIS = True
SENTIMENT_HISTORY_SIZE = 10  # Track last N sentiments
# ========================================


class MushroomImagePipeline:
    """Two-Stage image detection and classification pipeline (YOLO + ViT)"""
    
    def __init__(self, detector_path=None, classifier_path=None, mapping_file=None):
        """Initialize the image processing pipeline"""
        detector_path = detector_path or DETECTOR_PATH
        classifier_path = classifier_path or CLASSIFIER_PATH
        mapping_file = mapping_file or MAPPING_FILE
        
        print("  📸 Loading image processing models (YOLO + ViT)...")
        
        # Load species-toxicity mapping
        if Path(mapping_file).exists():
            import json
            with open(mapping_file, 'r') as f:
                self.species_toxicity = json.load(f)
            print(f"  ✅ Loaded {len(self.species_toxicity)} species mappings")
        else:
            print(f"  ⚠️  Mapping file not found: {mapping_file}")
            self.species_toxicity = {}
        
        # Load YOLO detector
        if not Path(detector_path).exists():
            print(f"  ⚠️  Detector not found: {detector_path}")
            self.detector = None
        else:
            self.detector = YOLO(detector_path)
            print(f"  ✅ YOLO detector loaded")
        
        # Load ViT classifier
        if not Path(classifier_path).exists():
            print(f"  ⚠️  Classifier not found: {classifier_path}")
            self.classifier = None
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            checkpoint = torch.load(classifier_path, map_location=self.device)
            
            # Get model configuration
            model_name = checkpoint.get('model_name', 'vit_b_16')
            num_classes = len(self.species_toxicity) if self.species_toxicity else 16
            
            # Create ViT model
            from torchvision import models
            import torch.nn as nn
            
            if model_name == 'vit_b_16':
                self.classifier = models.vit_b_16(weights=None)
            elif model_name == 'vit_b_32':
                self.classifier = models.vit_b_32(weights=None)
            elif model_name == 'vit_l_16':
                self.classifier = models.vit_l_16(weights=None)
            else:
                raise ValueError(f"Unknown model: {model_name}")
            
            # Check if this is an anti-overfitting version (with Dropout)
            state_dict = checkpoint['model_state_dict']
            has_dropout = any('heads.head.0' in k or 'heads.head.1' in k for k in state_dict.keys())
            
            if has_dropout:
                self.classifier.heads.head = nn.Sequential(
                    nn.Dropout(p=0.3),
                    nn.Linear(self.classifier.hidden_dim, num_classes)
                )
            else:
                self.classifier.heads.head = nn.Linear(self.classifier.hidden_dim, num_classes)
            
            # Load weights
            self.classifier.load_state_dict(checkpoint['model_state_dict'])
            self.classifier = self.classifier.to(self.device)
            self.classifier.eval()
            
            self.class_names = checkpoint.get('class_names', list(self.species_toxicity.keys()) if self.species_toxicity else [])
            print(f"  ✅ ViT classifier loaded ({model_name}, {num_classes} species)")
        
        # ViT classifier preprocessing (224x224)
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def process_image(self, image_path, conf_threshold=0.25):
        """
        Process image: detection + classification (YOLO + ViT)
        
        Returns:
            dict: dictionary containing detection results
        """
        if self.detector is None or self.classifier is None:
            return None
        
        # Read image
        img = cv2.imread(str(image_path))
        if img is None:
            return None
        
        # Stage 1: YOLO detect mushrooms
        results = self.detector.predict(
            source=str(image_path),
            conf=conf_threshold,
            verbose=False
        )
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                detections.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': conf
                })
        
        if len(detections) == 0:
            return {
                'num_mushrooms': 0,
                'mushrooms': []
            }
        
        # Stage 2: ViT classify each mushroom (identify species and toxicity)
        mushrooms = []
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det['bbox']
            cropped = img[y1:y2, x1:x2]
            
            if cropped.size == 0:
                continue
            
            # Convert to PIL Image
            mushroom_pil = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
            
            # Preprocessing
            mushroom_tensor = self.transform(mushroom_pil).unsqueeze(0).to(self.device)
            
            # ViT prediction
            with torch.no_grad():
                outputs = self.classifier(mushroom_tensor)
                probs = torch.softmax(outputs, dim=1)[0]
                
                # Top-3 predictions
                top3_probs, top3_indices = torch.topk(probs, k=min(3, len(self.class_names)))
                top3_probs = top3_probs.cpu().numpy()
                top3_indices = top3_indices.cpu().numpy()
                
                # Best prediction
                best_idx = int(top3_indices[0])
                best_species = self.class_names[best_idx]
                best_confidence = float(top3_probs[0])
                toxicity = self.species_toxicity.get(best_species, 'unknown')
            
            mushrooms.append({
                'id': i + 1,
                'species': best_species,
                'toxicity': toxicity,
                'confidence': best_confidence,
                'is_poisonous': (toxicity == 'poisonous' or toxicity == 'nonedible'),
                'top3_species': [self.class_names[int(idx)] for idx in top3_indices],
                'top3_confidences': [float(prob) for prob in top3_probs]
            })
        
        return {
            'num_mushrooms': len(mushrooms),
            'mushrooms': mushrooms
        }


class MushroomChatbot:
    """Integrated mushroom assistant: supports image processing and dialogue"""
    
    def __init__(self, use_rag=USE_RAG, enable_image=True, enable_sentiment=ENABLE_SENTIMENT_ANALYSIS):
        """Initialize chatbot with image processing"""
        self.use_rag = use_rag
        self.enable_image = enable_image
        self.enable_sentiment = enable_sentiment
        self.conversation_history = []
        
        # Sentiment tracking
        self.sentiment_history = deque(maxlen=SENTIMENT_HISTORY_SIZE)
        self.sentiment_analyzer = None
        
        self.load_model()
        
        # Load sentiment analyzer
        if self.enable_sentiment:
            try:
                print("  📊 Loading sentiment analyzer...")
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.sentiment_analyzer = pipeline(
                        "sentiment-analysis",
                        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                        device=0 if torch.cuda.is_available() else -1,
                        top_k=None  # Use top_k=None instead of return_all_scores=True to avoid deprecation warning
                    )
                print("  ✅ Sentiment analyzer loaded")
            except Exception as e:
                print(f"  ⚠️  Sentiment analyzer loading failed: {e}")
                print("  ⚠️  Continuing without sentiment analysis")
                self.enable_sentiment = False
        
        # Load image processing pipeline
        if self.enable_image:
            try:
                self.image_pipeline = MushroomImagePipeline()
            except Exception as e:
                print(f"  ⚠️  Image pipeline loading failed: {e}")
                self.enable_image = False
                self.image_pipeline = None
        else:
            self.image_pipeline = None
        
        if self.use_rag:
            print("🔍 Loading RAG knowledge base...")
            self.load_rag()
        
        print("✅ Initialization complete!\n")
    
    def load_model(self):
        """Load fine-tuned model with 4-bit quantization"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
        
        if USE_4BIT:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
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
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                base_model = AutoModelForCausalLM.from_pretrained(
                    BASE_MODEL,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    trust_remote_code=True
                )
        
        # Load LoRA adapter
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.model = PeftModel.from_pretrained(base_model, LORA_ADAPTER)
            print(f"  ✅ Loaded fine-tuned model: {LORA_ADAPTER}")
        except Exception as e:
            print(f"  ⚠️  Failed to load LoRA adapter: {e}")
            self.model = base_model
            print(f"  ⚠️  Using base model only")
        
        self.model.eval()
        
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            reserved = torch.cuda.memory_reserved(0) / 1024**3
            
    
    def load_rag(self):
        """Load RAG vector database"""
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
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
    
    def is_image_path(self, text: str) -> bool:
        """Check whether the input is an image path"""
        text = text.strip()
        
        # Check if this is an @image command
        if text.lower().startswith('@image'):
            return True
        
        # Check if this is a file path (supports relative and absolute paths)
        path = Path(text)
        # If this is a relative path, try to parse from the current working directory
        if not path.is_absolute():
            # Try to parse from the script's directory
            script_dir = Path(__file__).parent
            path = script_dir / text
        
        if path.exists() and path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']:
            return True
        return False
    
    def extract_image_path(self, text: str) -> str:
        """Extract image path from input"""
        text = text.strip()
        
        # If this is an @image command
        if text.lower().startswith('@image'):
            path = text[6:].strip()  # Remove '@image'
            if path.startswith('"') or path.startswith("'"):
                path = path[1:-1]  # Remove quotes
            return path
        
        # This is a direct path
        return text
    
    def should_use_rag(self, query: str) -> bool:
        """Check if RAG should be used"""
        if not self.use_rag:
            return False
        
        query_lower = query.lower().strip()
        search_triggers = ['@search', '/search', 'search:']
        return any(query_lower.startswith(trigger) for trigger in search_triggers)
    
    def extract_search_query(self, query: str) -> str:
        """Extract search query"""
        query_stripped = query.strip()
        for trigger in ['@search', '/search', 'search:']:
            if query_stripped.lower().startswith(trigger):
                return query_stripped[len(trigger):].strip()
        return query
    
    def retrieve_context(self, query: str) -> str:
        """Retrieve context from RAG database"""
        if not self.use_rag:
            return ""
        
        try:
            results = self.rag_collection.query(
                query_texts=[query],
                n_results=TOP_K
            )
            
            context_parts = []
            for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0]), 1):
                topic = metadata.get('topic', 'Unknown')
                # Increase document length to include more visual feature information
                doc_preview = doc[:500] if len(doc) > 500 else doc
                context_parts.append(f"[Source {i}: {topic}]\n{doc_preview}")
            
            return "\n\n".join(context_parts)
        except Exception as e:
            print(f"  ⚠️  RAG retrieval failed: {e}")
            return ""
    
    def format_image_results(self, results: dict) -> str:
        """Format image detection results into text"""
        if results is None or results['num_mushrooms'] == 0:
            return "No mushrooms detected."
        
        text = f"Detected {results['num_mushrooms']} mushrooms:\n\n"
        
        for mushroom in results['mushrooms']:
            species = mushroom.get('species', 'Unknown')
            toxicity = mushroom.get('toxicity', 'unknown')
            confidence = mushroom['confidence']
            
            if mushroom['is_poisonous']:
                status = "⚠️ Poisonous"
            elif toxicity == 'edible':
                status = "✅ Edible"
            else:
                status = "❓ Unknown"
            
            text += f"Mushroom {mushroom['id']}: {species}\n"
            text += f"  Toxicity: {status} ({toxicity})\n"
            text += f"  Confidence: {confidence:.2%}\n"
            
            # Show Top-3 predictions if available
            if 'top3_species' in mushroom and len(mushroom['top3_species']) > 1:
                text += f"  Other possibilities: "
                for i, (sp, conf) in enumerate(zip(mushroom['top3_species'][1:], mushroom['top3_confidences'][1:]), 1):
                    text += f"{sp}({conf:.1%})"
                    if i < len(mushroom['top3_species']) - 1:
                        text += ", "
                text += "\n"
            text += "\n"
        
        return text
    
    def generate_response(self, user_input: str, image_results: dict = None) -> str:
        
        # If there are image results, format them first
        image_context = ""
        rag_context = ""
        
        if image_results:
            image_context = self.format_image_results(image_results)
            
            # Automatically search RAG knowledge base for detected mushroom species
            detected_species = []
            for mushroom in image_results.get('mushrooms', []):
                species = mushroom.get('species', '')
                if species and species not in detected_species:
                    detected_species.append(species)
            
            # Search for detailed information for each detected species
            if detected_species and self.use_rag:
                print("  🔍 [Auto-searching knowledge base for detected species...]")
                species_contexts = []
                for species in detected_species:
                    # Try multiple search methods
                    search_queries = [
                        species,
                        species.replace('-', ' '),  # Amanita-calyptroderma -> Amanita calyptroderma
                        species.split('-')[0] if '-' in species else species,  # Search genus only
                    ]
                    
                    for query in search_queries:
                        species_info = self.retrieve_context(query)
                        if species_info:
                            species_contexts.append(f"Information about {species}:\n{species_info}")
                            break  # Stop once information is found
                
                if species_contexts:
                    rag_context = "\n\n".join(species_contexts)
                    print(f"  ✅ Found information for {len(species_contexts)} species")
                else:
                    print(f"  ⚠️  No RAG information found for detected species")
        
        # Check if RAG should be manually triggered
        use_rag_manual = self.should_use_rag(user_input)
        
        if use_rag_manual:
            print("  🔍 [Searching wiki knowledge base...]")
            search_query = self.extract_search_query(user_input)
            manual_context = self.retrieve_context(search_query)
            if manual_context:
                rag_context = f"{rag_context}\n\nAdditional context:\n{manual_context}" if rag_context else manual_context
            actual_query = search_query
        else:
            actual_query = user_input
            # If an image was processed but no RAG context was found, remind user they can use @search
            if image_context and not rag_context:
                print("  💡 Tip: Use '@search <species>' to get detailed information about detected mushrooms")
        
        # Analyze sentiment if enabled
        current_sentiment = None
        sentiment_info = ""
        if self.enable_sentiment and not self.is_image_path(actual_query):
            current_sentiment = self.analyze_sentiment(actual_query)
            self.sentiment_history.append(current_sentiment)
            
            # Print reasoning process before generating response
            print("\n" + "=" * 80)
            print(current_sentiment.get('reasoning', 'Sentiment analysis reasoning not available'))
            print("=" * 80 + "\n")
            
            sentiment_info = f"\n\n[Sentiment Analysis]\nCurrent message sentiment: {current_sentiment['label'].upper()} (confidence: {current_sentiment['score']:.2%}, happiness score: {current_sentiment.get('happiness_score', 5):.1f}/10)"
            if len(self.sentiment_history) > 1:
                sentiment_info += f"\n{self.get_sentiment_summary()}"
        
        # Build prompt
        if image_context:
            # There are image detection results - require visual feature evidence
            base_system = """You are a knowledgeable mushroom expert assistant. You have just analyzed an image and detected mushrooms. 

IMPORTANT: When explaining your identification, you MUST provide specific visual evidence from the image analysis, such as:
- Cap color, shape, and texture
- Stem characteristics
- Gills or pores
- Size and overall appearance
- Any distinctive features

Explain WHY you believe the mushroom is edible or poisonous based on these visual characteristics. Be specific and detailed."""
            
            # Add sentiment awareness to system message
            if sentiment_info:
                system_message = base_system + "\n\n" + sentiment_info + "\n\nPlease adjust your tone appropriately based on the user's sentiment. If the user seems concerned or negative, be more reassuring and careful. If positive, you can be more enthusiastic."
            else:
                system_message = base_system
            
            messages = [
                {"role": "system", "content": system_message},
            ]
            
            for msg in self.conversation_history[-4:]:
                messages.append(msg)
            
            user_message = f"""Image Analysis Results:
{image_context}

User question: {actual_query}"""
            
            if rag_context:
                user_message = f"""Context from knowledge base (including visual characteristics and identification features):
{rag_context}

{user_message}

Please provide a detailed explanation with visual evidence for why each mushroom is identified as edible or poisonous."""
            
            messages.append({"role": "user", "content": user_message})
        
        elif rag_context:
            # Only RAG context
            base_system = """You are a knowledgeable mushroom expert assistant. Use the provided context to answer the user's question accurately. When discussing mushroom identification, always mention specific visual characteristics (cap color, stem, gills, size, etc.) as evidence."""
            
            # Add sentiment awareness
            if sentiment_info:
                system_message = base_system + "\n\n" + sentiment_info + "\n\nPlease adjust your tone appropriately based on the user's sentiment."
            else:
                system_message = base_system
            
            messages = [
                {"role": "system", "content": system_message},
            ]
            
            for msg in self.conversation_history[-4:]:
                messages.append(msg)
            
            user_message = f"""Context from knowledge base:
{rag_context}

User question: {actual_query}"""
            
            messages.append({"role": "user", "content": user_message})
        
        else:
            # Normal conversation
            base_system = "You are a helpful and friendly mushroom expert assistant."
            
            # Add sentiment awareness
            if sentiment_info:
                system_message = base_system + "\n\n" + sentiment_info + "\n\nPlease adjust your tone appropriately based on the user's sentiment. Be empathetic and supportive if the user seems concerned or negative."
            else:
                system_message = base_system
            
            messages = [
                {"role": "system", "content": system_message}
            ]
            
            for msg in self.conversation_history[-4:]:
                messages.append(msg)
            
            messages.append({"role": "user", "content": actual_query})
        
        # Generate response
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Display progress prompt
        print("  🤖 [Generating response...]")
        
        # Suppress transformers warnings during generation
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with torch.no_grad():
                # Suppress specific transformers warnings by setting environment
                import os
                old_warn = os.environ.get('TRANSFORMERS_VERBOSITY', '')
                os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
                try:
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=MAX_NEW_TOKENS,
                        temperature=TEMPERATURE,
                        do_sample=True,
                        top_p=0.9,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                finally:
                    if old_warn:
                        os.environ['TRANSFORMERS_VERBOSITY'] = old_warn
                    elif 'TRANSFORMERS_VERBOSITY' in os.environ:
                        del os.environ['TRANSFORMERS_VERBOSITY']
        
        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return response.strip()
    
    def chat(self, user_input: str) -> str:
        """Process user input and return response"""
        # Check if this is a image path
        image_results = None
        image_path = None
        
        if self.enable_image and self.image_pipeline and self.is_image_path(user_input):
            # Extract image path
            image_path = self.extract_image_path(user_input)
            
            # Process relative path
            path = Path(image_path)
            if not path.is_absolute():
                # Try to parse from the script's directory
                script_dir = Path(__file__).parent
                path = script_dir / image_path
                if not path.exists():
                    # Try to parse from the current working directory
                    path = Path(image_path).resolve()
            
            if not path.exists():
                print(f"  ⚠️  Image path does not exist: {image_path}")
                return f"Sorry, cannot find image: {image_path}. Please check that the path is correct."
            
            print(f"  📸 [Processing image: {path.name}...]")
            try:
                image_results = self.image_pipeline.process_image(str(path))
                if image_results:
                    print(f"  ✅ Detected {image_results['num_mushrooms']} mushroom(s)")
                    # Display detection results
                    print(self.format_image_results(image_results))
            except Exception as e:
                print(f"  ⚠️  Image processing failed: {e}")
                import traceback
                traceback.print_exc()
                image_results = None
        
        # Generate response
        # If there are image results, convert user input to a question about the image
        if image_results:
            # If the user only entered the image path, default to asking "Are these mushrooms safe to eat?"
            if image_path and (user_input.strip() == image_path or user_input.strip().startswith('@image')):
                actual_query = "Please analyze these mushrooms in detail, explain whether they are safe to eat, and provide visual evidence such as cap color, stem characteristics, and gills."
            else:
                actual_query = user_input
        else:
            actual_query = user_input
        
        response = self.generate_response(actual_query, image_results)
        
        # Update conversation history
        if self.should_use_rag(actual_query):
            clean_query = self.extract_search_query(actual_query)
        else:
            clean_query = actual_query
        
        self.conversation_history.append({"role": "user", "content": clean_query})
        self.conversation_history.append({"role": "assistant", "content": response})
        
        return response
    
    def analyze_sentiment(self, text: str) -> Dict[str, any]:
        """
        Analyze sentiment of user message using LLM to reason about emotion and score
        
        Returns:
            dict with 'label', 'score', 'happiness_score' (0-10), and 'reasoning'
        """
        if not self.enable_sentiment:
            return {
                'label': 'neutral', 
                'score': 0.0, 
                'happiness_score': 5,
                'reasoning': 'Sentiment analysis disabled'
            }
        
        try:
            # Use LLM to analyze sentiment and reason about emotion
            text_short = text[:512]
            
            # Create prompt for sentiment analysis
            sentiment_prompt = f"""Analyze the emotional sentiment of the following message and provide:
1. Primary sentiment: negative, neutral, or positive
2. Confidence level (0.0 to 1.0)
3. Happiness score (0-10, where 0=very sad/unhappy, 5=neutral, 10=very happy)
4. Detailed reasoning process

Message: "{text_short}"

Please respond in the following JSON format:
{{
    "sentiment": "negative/neutral/positive",
    "confidence": 0.0-1.0,
    "happiness_score": 0-10,
    "reasoning": "detailed explanation of your analysis"
}}"""

            messages = [
                {"role": "system", "content": "You are an expert at analyzing emotional sentiment. Analyze the user's message and provide sentiment analysis with detailed reasoning."},
                {"role": "user", "content": sentiment_prompt}
            ]
            
            # Generate response
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=256,
                        temperature=0.3,
                        do_sample=True,
                        top_p=0.9,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
            
            response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            
            # Parse JSON response
            import json
            import re
            
            # Try to extract JSON from response
            json_match = re.search(r'\{[^{}]*"sentiment"[^{}]*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                try:
                    result = json.loads(json_str)
                    label = result.get('sentiment', 'neutral').lower()
                    confidence = float(result.get('confidence', 0.5))
                    happiness_score = float(result.get('happiness_score', 5))
                    reasoning_text = result.get('reasoning', 'No reasoning provided')
                    
                    # Clamp values
                    confidence = max(0.0, min(1.0, confidence))
                    happiness_score = max(0, min(10, round(happiness_score, 1)))
                    
                    # Generate formatted reasoning
                    reasoning = self._format_llm_reasoning(text_short, label, confidence, happiness_score, reasoning_text)
                    
                    return {
                        'label': label,
                        'score': confidence,
                        'happiness_score': happiness_score,
                        'reasoning': reasoning
                    }
                except json.JSONDecodeError:
                    pass
            
            # Fallback: try to extract values from text if JSON parsing fails
            label_match = re.search(r'sentiment["\']?\s*[:=]\s*["\']?(negative|neutral|positive)', response, re.IGNORECASE)
            score_match = re.search(r'happiness_score["\']?\s*[:=]\s*(\d+\.?\d*)', response, re.IGNORECASE)
            conf_match = re.search(r'confidence["\']?\s*[:=]\s*(\d+\.?\d*)', response, re.IGNORECASE)
            
            label = label_match.group(1).lower() if label_match else 'neutral'
            happiness_score = float(score_match.group(1)) if score_match else 5.0
            confidence = float(conf_match.group(1)) if conf_match else 0.5
            
            happiness_score = max(0, min(10, round(happiness_score, 1)))
            confidence = max(0.0, min(1.0, confidence))
            
            reasoning = self._format_llm_reasoning(text_short, label, confidence, happiness_score, response)
            
            return {
                'label': label,
                'score': confidence,
                'happiness_score': happiness_score,
                'reasoning': reasoning
            }
            
        except Exception as e:
            print(f"  ⚠️  Sentiment analysis error: {e}")
            import traceback
            traceback.print_exc()
            return {
                'label': 'neutral', 
                'score': 0.0,
                'happiness_score': 5,
                'reasoning': f'Analysis failed: {str(e)}'
            }
    
    def _format_llm_reasoning(self, text: str, label: str, confidence: float, happiness_score: float, reasoning_text: str) -> str:
        """Format LLM reasoning into structured output"""
        reasoning_parts = []
        reasoning_parts.append(f"📊 Sentiment Analysis Reasoning Process:")
        reasoning_parts.append(f"")
        reasoning_parts.append(f"1️⃣ Text Analysis:")
        reasoning_parts.append(f"   - Input text: \"{text}\"")
        reasoning_parts.append(f"")
        reasoning_parts.append(f"2️⃣ Model Reasoning:")
        reasoning_parts.append(f"   {reasoning_text}")
        reasoning_parts.append(f"")
        reasoning_parts.append(f"3️⃣ Final Results:")
        reasoning_parts.append(f"   - Primary sentiment: {label.upper()}")
        reasoning_parts.append(f"   - Confidence: {confidence:.1%}")
        reasoning_parts.append(f"")
        reasoning_parts.append(f"4️⃣ Final Score: {happiness_score:.1f}/10")
        
        # Determine emotion state
        if happiness_score < 2:
            emoji = "😢"
            state = "Very sad"
        elif happiness_score < 4:
            emoji = "😟"
            state = "Sad"
        elif happiness_score < 6:
            emoji = "😐"
            state = "Neutral"
        elif happiness_score < 8:
            emoji = "🙂"
            state = "Happy"
        else:
            emoji = "😊"
            state = "Very happy"
        
        reasoning_parts.append(f"   {emoji} Emotional state: {state}")
        
        return "\n".join(reasoning_parts)
    
    def _generate_sentiment_reasoning(self, text: str, label: str, confidence: float, happiness_score: float, all_scores: Dict[str, float]) -> str:
        """Generate detailed reasoning process for sentiment analysis"""
        # Analyze text characteristics
        text_lower = text.lower()
        
        # Detect emotional indicators
        positive_words = ['good', 'great', 'excellent', 'wonderful', 'happy', 'love', 'like', 'thanks', 'thank', 'helpful', 'amazing', 'perfect', 'nice', 'awesome']
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'disappointed', 'worried', 'concerned', 'scared', 'afraid', 'problem', 'error', 'wrong', 'sad', 'angry']
        question_words = ['?', 'what', 'how', 'why', 'when', 'where', 'which', 'who']
        
        detected_positive = sum(1 for word in positive_words if word in text_lower)
        detected_negative = sum(1 for word in negative_words if word in text_lower)
        is_question = any(qw in text_lower for qw in question_words)
        
        # Build reasoning
        reasoning_parts = []
        reasoning_parts.append(f"📊 Sentiment Analysis Reasoning Process:")
        reasoning_parts.append(f"")
        
        # Text analysis
        reasoning_parts.append(f"1️⃣ Text Feature Analysis:")
        if detected_positive > 0:
            reasoning_parts.append(f"   - Detected {detected_positive} positive word(s)")
        if detected_negative > 0:
            reasoning_parts.append(f"   - Detected {detected_negative} negative word(s)")
        if is_question:
            reasoning_parts.append(f"   - Contains question, likely inquiry-based")
        if not detected_positive and not detected_negative:
            reasoning_parts.append(f"   - No obvious emotional words detected, leaning neutral")
        
        # Model prediction
        reasoning_parts.append(f"")
        reasoning_parts.append(f"2️⃣ Model Prediction Results:")
        reasoning_parts.append(f"   - Primary sentiment: {label.upper()}")
        reasoning_parts.append(f"   - Confidence: {confidence:.1%}")
        reasoning_parts.append(f"   - Sentiment scores breakdown:")
        for sent_label, score in sorted(all_scores.items(), key=lambda x: x[1], reverse=True):
            bar_length = int(score * 20)
            bar = "█" * bar_length + "░" * (20 - bar_length)
            reasoning_parts.append(f"     • {sent_label:8s}: {score:.1%} {bar}")
        
        # Happiness score calculation
        reasoning_parts.append(f"")
        reasoning_parts.append(f"3️⃣ Happiness Score Calculation:")
        if label == 'negative':
            reasoning_parts.append(f"   - Sentiment type: Negative (score range: 0-2)")
            reasoning_parts.append(f"   - Formula: 2 × (1 - confidence)² (conservative)")
            calc_step = 2 * ((1 - confidence) ** 2)
            reasoning_parts.append(f"   - Calculation: 2 × (1 - {confidence:.2f})² = {calc_step:.2f} ≈ {happiness_score:.1f}")
        elif label == 'neutral':
            reasoning_parts.append(f"   - Sentiment type: Neutral (score range: 3-6)")
            reasoning_parts.append(f"   - Formula: 3.5 + 1.5 × confidence (conservative)")
            calc_step = 3.5 + 1.5 * confidence
            reasoning_parts.append(f"   - Calculation: 3.5 + 1.5 × {confidence:.2f} = {calc_step:.2f} ≈ {happiness_score:.1f}")
        else:  # positive
            reasoning_parts.append(f"   - Sentiment type: Positive (score range: 7-10)")
            reasoning_parts.append(f"   - Formula: 7 + 2.5 × confidence^1.2 (conservative)")
            calc_step = 7 + 2.5 * (confidence ** 1.2)
            reasoning_parts.append(f"   - Calculation: 7 + 2.5 × {confidence:.2f}^1.2 = {calc_step:.2f} ≈ {happiness_score:.1f}")
        
        # Final score
        reasoning_parts.append(f"")
        reasoning_parts.append(f"4️⃣ Final Score: {happiness_score:.1f}/10")
        emoji_map = {
            (0, 2): "😢",  # Very sad
            (2, 4): "😟",  # Sad
            (4, 6): "😐",  # Neutral
            (6, 8): "🙂",  # Happy
            (8, 10): "😊"  # Very happy
        }
        for (low, high), emoji in emoji_map.items():
            if low <= happiness_score < high or (high == 10 and happiness_score == 10):
                emotion_state = "Very sad" if happiness_score < 2 else "Sad" if happiness_score < 4 else "Neutral" if happiness_score < 6 else "Happy" if happiness_score < 8 else "Very happy"
                reasoning_parts.append(f"   {emoji} Emotional state: {emotion_state}")
                break
        
        return "\n".join(reasoning_parts)
    
    def get_sentiment_summary(self) -> str:
        """Get summary of sentiment changes across session"""
        if not self.sentiment_history or len(self.sentiment_history) == 0:
            return "No sentiment data available yet."
        
        # Count sentiments
        sentiment_counts = {'positive': 0, 'neutral': 0, 'negative': 0}
        avg_happiness = 0.0
        for sent in self.sentiment_history:
            sentiment_counts[sent['label']] += 1
            avg_happiness += sent.get('happiness_score', 5)
        
        total = len(self.sentiment_history)
        avg_happiness = avg_happiness / total if total > 0 else 5.0
        
        summary = f"Sentiment trend (last {total} messages): "
        summary += f"Positive: {sentiment_counts['positive']}, "
        summary += f"Neutral: {sentiment_counts['neutral']}, "
        summary += f"Negative: {sentiment_counts['negative']}"
        summary += f" | Average happiness: {avg_happiness:.1f}/10"
        
        # Detect trend
        if len(self.sentiment_history) >= 3:
            recent = [s['label'] for s in list(self.sentiment_history)[-3:]]
            if all(s == 'positive' for s in recent):
                summary += " (Trending positive 📈)"
            elif all(s == 'negative' for s in recent):
                summary += " (Trending negative 📉)"
        
        return summary
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        self.sentiment_history.clear()
        print("🗑️  Conversation history cleared")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    print("=" * 80)
    print("🍄 Integrated Mushroom Assistant")
    print("   Image Detection + Classification + RAG Chatbot")
    print("=" * 80)
    print("\n💡 Commands:")
    print("  - Ask questions normally for quick answers")
    print("  - Use '@search <query>' to search the wiki knowledge base")
    print("  - Analyze image using one of these methods:")
    print("    1. Enter image path directly:")
    print("       Example: ../../Dataset/merged_mushroom_dataset/test/images/sample.jpg")
    print("    2. Use '@image <path>' command:")
    print("       Example: @image sample.jpg")
    print("    3. Drag and drop image file into terminal (Windows)")
    print("  - 'clear' to clear conversation history")
    print("  - 'quit' or 'exit' to quit")
    print()
    print("⚙️  Features:")
    print("  ✅ Two-Stage Pipeline (YOLO + ViT)")
    print("  ✅ Species Classification (16 species)")
    print("  ✅ RAG Knowledge Base")
    print("  ✅ Sentiment Analysis (tracks user sentiment across sessions)")
    print()
    
    # Initialize chatbot
    bot = MushroomChatbot(use_rag=USE_RAG, enable_image=True)
    
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
        
        except torch.cuda.OutOfMemoryError:
            print("\n❌ GPU memory is insufficient! Try the following:")
            print("  1. Use 'clear' to reset conversation history")
            print("  2. Restart the program")
            print("  3. Close other programs using the GPU")
            bot.clear_history()
        except Exception as e:
            print(f"\n❌ Error: {e}\n")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()

