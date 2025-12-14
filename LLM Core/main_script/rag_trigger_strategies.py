"""
Alternative RAG Triggering Strategies

This file contains different approaches you can use in chat_with_rag.py
Choose the one that fits your needs best.
"""

# ============================================================
# OPTION 1: Keyword-Based (Current Implementation)
# ============================================================
# Pros: Fast, simple, predictable
# Cons: May miss some relevant queries, may trigger on irrelevant ones

def should_use_rag_keywords(query: str) -> bool:
    """Current implementation - keyword matching"""
    # See main implementation in chat_with_rag.py
    pass


# ============================================================
# OPTION 2: Confidence-Based (Advanced)
# ============================================================
# Pros: More accurate, adapts to query relevance
# Cons: Slower (requires actual RAG query), more complex

def should_use_rag_confidence(self, query: str, threshold: float = 0.5) -> bool:
    """
    Use RAG if the top retrieved document has high enough similarity.
    
    Args:
        query: User's question
        threshold: Minimum similarity score (0-1) to trigger RAG
                  - 0.3-0.4: Very permissive (use RAG often)
                  - 0.5-0.6: Balanced (recommended)
                  - 0.7+: Conservative (only high-confidence matches)
    """
    if not self.use_rag:
        return False
    
    # Filter out casual conversation first
    casual_phrases = ['hello', 'hi ', 'hey', 'thanks', 'bye']
    if any(phrase in query.lower() for phrase in casual_phrases):
        return False
    
    try:
        # Do a quick retrieval
        results = self.rag_collection.query(
            query_texts=[query],
            n_results=1
        )
        
        # Check if we have results and distances
        if results and 'distances' in results and results['distances']:
            # ChromaDB returns L2 distance (lower is better)
            # Convert to similarity: similarity = 1 / (1 + distance)
            distance = results['distances'][0][0]
            similarity = 1 / (1 + distance)
            
            print(f"  📊 RAG confidence: {similarity:.2%}")
            return similarity >= threshold
        
        return False
        
    except Exception as e:
        print(f"  ⚠️  Confidence check failed: {e}")
        return False


# ============================================================
# OPTION 3: Hybrid Approach (Best of Both Worlds)
# ============================================================
# Pros: Combines speed of keywords with accuracy of confidence
# Cons: Most complex

def should_use_rag_hybrid(self, query: str) -> bool:
    """
    Hybrid approach:
    1. First check keywords (fast filter)
    2. If keywords match, verify with confidence score
    """
    if not self.use_rag:
        return False
    
    query_lower = query.lower()
    
    # === Step 1: Quick Filters ===
    
    # Exclude casual conversation
    casual_phrases = ['hello', 'hi ', 'hey', 'thanks', 'bye', 'how are you']
    if any(phrase in query_lower for phrase in casual_phrases):
        return False
    
    # Check for mushroom-specific terms (high confidence triggers)
    high_confidence_keywords = [
        'mushroom', 'fungi', 'fungus', 'amanita', 'psilocybin',
        'poisonous', 'toxic', 'edible', 'deadly'
    ]
    if any(keyword in query_lower for keyword in high_confidence_keywords):
        return True  # Skip confidence check for obvious mushroom queries
    
    # === Step 2: Check for Question Patterns ===
    question_patterns = [
        'what is', 'what are', 'how to', 'where', 'when',
        'can i', 'should i', 'is it', 'tell me about'
    ]
    
    has_question_pattern = any(pattern in query_lower for pattern in question_patterns)
    
    if not has_question_pattern:
        return False  # Not a factual question
    
    # === Step 3: Verify with Confidence Score ===
    try:
        results = self.rag_collection.query(
            query_texts=[query],
            n_results=1
        )
        
        if results and 'distances' in results and results['distances']:
            distance = results['distances'][0][0]
            similarity = 1 / (1 + distance)
            
            # Lower threshold since we already filtered by keywords
            return similarity >= 0.4
        
        return False
        
    except:
        # If confidence check fails, fall back to keyword matching
        return has_question_pattern


# ============================================================
# OPTION 4: Always Use RAG (Simple)
# ============================================================
# Pros: Never miss relevant information
# Cons: Slower, may retrieve irrelevant context for casual chat

def should_use_rag_always(self, query: str) -> bool:
    """Always use RAG except for greetings"""
    if not self.use_rag:
        return False
    
    # Only exclude obvious greetings
    greetings = ['hello', 'hi', 'hey', 'bye', 'goodbye']
    return not any(greeting == query.lower().strip() for greeting in greetings)


# ============================================================
# OPTION 5: Manual Toggle (User Control)
# ============================================================
# Pros: User has full control
# Cons: Requires user to know when to use RAG

def should_use_rag_manual(self, query: str) -> bool:
    """
    Let user control RAG with special syntax:
    - Normal query: "What is Amanita?" -> No RAG
    - With @rag: "@rag What is Amanita?" -> Use RAG
    """
    return query.strip().startswith('@rag')

# If using this, strip @rag from query before processing:
# if query.startswith('@rag'):
#     query = query[4:].strip()


# ============================================================
# RECOMMENDATION
# ============================================================
"""
For your mushroom chatbot, I recommend:

1. START WITH: Option 1 (Keyword-Based) - Current implementation
   - Good balance of speed and accuracy
   - Easy to debug and tune
   
2. IF YOU WANT BETTER ACCURACY: Option 3 (Hybrid)
   - Best overall performance
   - Prevents false positives
   
3. IF SPEED IS CRITICAL: Keep Option 1
   
4. IF YOU WANT TO EXPERIMENT: Add Option 5 (Manual Toggle)
   - Users can use @rag when they want knowledge base
   - Falls back to keyword matching otherwise

Example usage in chat_with_rag.py:
    # Replace the should_use_rag method with your chosen implementation
    # Adjust thresholds based on your testing
"""
