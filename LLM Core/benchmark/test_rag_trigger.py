#!/usr/bin/env python3
"""
Test RAG triggering logic with various queries
"""

# Simulate the should_use_rag method
def should_use_rag(query: str) -> bool:
    """Current implementation from chat_with_rag.py"""
    query_lower = query.lower()
    
    # === Strategy 1: Factual Question Patterns ===
    factual_patterns = [
        'what is', 'what are', 'what does', 'what\'s',
        'how to', 'how do', 'how can', 'how does',
        'where', 'when', 'why',
        'tell me about', 'explain', 'describe',
        'which', 'who',
    ]
    
    # === Strategy 2: Question Words ===
    question_starters = ['is it', 'are they', 'can i', 'should i', 'could i', 'would']
    
    # === Strategy 3: Mushroom-Specific Keywords ===
    mushroom_keywords = [
        'mushroom', 'fungi', 'fungus', 'mycelium', 'spore',
        'poisonous', 'edible', 'toxic', 'safe', 'deadly',
        'identify', 'identification', 'look like', 'appearance',
        'found', 'grow', 'habitat', 'season',
        'eat', 'consume', 'cook', 'prepare',
        'amanita', 'psilocybin', 'chanterelle', 'morel',
        'cap', 'stem', 'gill', 'ring', 'volva'
    ]
    
    # === Strategy 4: Exclude Casual Conversation ===
    casual_phrases = [
        'hello', 'hi ', 'hey', 'good morning', 'good afternoon',
        'how are you', 'thanks', 'thank you', 'bye', 'goodbye',
        'nice to meet', 'pleased to meet',
        'what\'s up', 'whats up', 'sup',  # Common greetings
        'tell me a joke', 'make me laugh'  # Entertainment requests
    ]
    
    # Don't use RAG for casual greetings
    if any(phrase in query_lower for phrase in casual_phrases):
        return False
    
    # Use RAG if it matches factual patterns OR question starters OR mushroom keywords
    has_factual_pattern = any(pattern in query_lower for pattern in factual_patterns)
    has_question_starter = any(starter in query_lower for starter in question_starters)
    has_mushroom_keyword = any(keyword in query_lower for keyword in mushroom_keywords)
    
    # Trigger RAG if any condition is met
    return has_factual_pattern or has_question_starter or has_mushroom_keyword


# Test queries
test_queries = [
    # Should trigger RAG (factual questions)
    "What is Amanita muscaria?",
    "How to identify poisonous mushrooms?",
    "Tell me about morel mushrooms",
    "Where do chanterelles grow?",
    "Can I eat this mushroom?",
    "Is it safe to consume psilocybin?",
    
    # Should trigger RAG (mushroom keywords)
    "I found a red mushroom with white spots",
    "This fungus has a ring on the stem",
    "Looking for edible mushrooms in my area",
    
    # Should NOT trigger RAG (casual conversation)
    "Hello!",
    "Hi there",
    "Thanks for your help",
    "How are you doing?",
    "Good morning",
    "What's up?",
    "Tell me a joke",
    
    # Should NOT trigger RAG (general chat)
    "That's interesting",
    "I see",
    "What's the weather like?",
    
    # Edge cases
    "What's the difference between fungi and plants?",  # Should trigger (has 'what' + 'fungi')
    "How are mushrooms classified?",  # Should trigger (has 'how' + 'mushroom')
]

print("=" * 80)
print("🧪 Testing RAG Trigger Logic")
print("=" * 80)
print()

# Test each query
rag_count = 0
no_rag_count = 0

for query in test_queries:
    use_rag = should_use_rag(query)
    icon = "🔍" if use_rag else "💬"
    status = "USE RAG" if use_rag else "NO RAG"
    print(f"{icon} [{status:8}] {query}")
    
    if use_rag:
        rag_count += 1
    else:
        no_rag_count += 1

print()
print("=" * 80)
print(f"Summary: {rag_count} queries will use RAG, {no_rag_count} will not")
print()
print("Legend:")
print("  🔍 = Will use RAG (retrieve from knowledge base)")
print("  💬 = Will use model only (no RAG)")
print("=" * 80)
