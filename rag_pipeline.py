import os
from typing import Dict, Optional, List
from langchain_groq import ChatGroq
import streamlit as st
from dotenv import load_dotenv
import logging
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# ============================================================================
# LLM INITIALIZATION
# ============================================================================

def create_groq_llm() -> ChatGroq:
    """Create and return a Groq LLM instance."""
    api_key = os.getenv("GROQ_API_KEY")
    
    if not api_key:
        raise ValueError("❌ GROQ_API_KEY not found in environment variables")
    
    try:
        llm = ChatGroq(
            model="openai/gpt-oss-120b",  # Using a known working model
            temperature=0.3,  # Balanced for focused yet natural responses
            max_tokens=600,  # Reasonable limit for concise but complete output
            groq_api_key=api_key,
            max_retries=2,
            timeout=30,
        )
        logger.info("✅ Groq LLM initialized successfully")
        return llm
    except Exception as e:
        logger.error(f"❌ Failed to create Groq LLM: {e}")
        raise

# ============================================================================
# INGREDIENT PARSING
# ============================================================================

def parse_ingredients(ingredient_text: str) -> List[str]:
    """Parse ingredient text into individual ingredients."""
    # Split by comma and parentheses
    ingredients = re.split(r'[,()]', ingredient_text)
    
    # Clean up and filter
    cleaned = []
    for ing in ingredients:
        ing = ing.strip()
        if ing and len(ing) > 1 and not ing.isdigit():
            cleaned.append(ing)
    
    return cleaned

# ============================================================================
# CONCISE EXPLANATION GENERATOR
# ============================================================================

class IngredientExplainer:
    """Generates ultra-concise ingredient safety summaries."""
    
    def __init__(self, llm: ChatGroq):
        """Initialize the explainer."""
        self.llm = llm
    
    def generate_explanation(self, ingredients: str, risk_level: int, risk_category: str) -> str:
        """Generate EXTREMELY concise explanation for ingredients, maximum 5 lines for each ingredient, NOTHING MORE.
        
        Args:
            ingredients: The ingredient list
            risk_level: The classified risk level (1-5)
            risk_category: The risk category
        
        Returns:
            str: Ultra-concise explanation (max 1200 tokens / ~300-400 words)
        """
        try:
            # Parse ingredients
            ingredient_list = parse_ingredients(ingredients)
            
            # Create concise but comprehensive prompt
            num_ingredients = len(ingredient_list)
            words_per_ingredient = min(50, max(20, 400 // max(num_ingredients, 1)))
            
            # Simplified prompt to ensure we get a response
            prompt = f"""Food safety expert: Briefly explain these ingredients ({", ".join(ingredient_list)}) at risk level {risk_level} ({risk_category}). 
For each ingredient: name, purpose, concern, safer option if any. Be concise."""
            
            response = self.llm.invoke(prompt)
            
            if hasattr(response, "content"):
                result = response.content.strip()
            else:
                result = str(response).strip()
            
            logger.info("✅ Concise explanation generated successfully")
            return result
            
        except Exception as e:
            error_msg = f"Could not generate explanation: {str(e)}"
            logger.error(f"❌ Error generating explanation: {error_msg}")
            return error_msg

# ============================================================================
# PIPELINE INITIALIZATION
# ============================================================================

def initialize_rag_pipeline() -> Optional[IngredientExplainer]:
    """Initialize the explanation pipeline."""
    try:
        logger.info("🔄 Initializing ingredient explainer...")
        llm = create_groq_llm()
        explainer = IngredientExplainer(llm)
        logger.info("✅ Ingredient explainer initialized successfully")
        return explainer
        
    except Exception as e:
        error_msg = f"Failed to initialize RAG pipeline: {str(e)}"
        logger.error(f"❌ Error initializing explainer: {error_msg}")
        return None

# ============================================================================
# CALL RAG PIPELINE
# ============================================================================

def call_rag_pipeline(
    explainer: IngredientExplainer,
    user_input: str,
    classification_result: Optional[Dict] = None
) -> str:
    """Generate explanation for ingredients.
    
    Args:
        explainer: The explainer instance
        user_input: The ingredient list
        classification_result: Classification data
    
    Returns:
        str: Ultra-concise explanation
    """
    try:
        if not classification_result:
            return "No classification data available for explanation."
        
        risk_level = classification_result.get("risk_level", "unknown")
        risk_category = classification_result.get("risk_category", "unknown")
        
        explanation = explainer.generate_explanation(user_input, risk_level, risk_category)
        logger.info("✅ Explanation generated successfully")
        return explanation
        
    except Exception as e:
        error_msg = f"Error in RAG pipeline: {str(e)}"
        logger.error(f"❌ Error calling explainer: {error_msg}")
        return error_msg

# ============================================================================
# SESSION MEMORY (minimal)
# ============================================================================

def init_session_memory() -> None:
    """Initialize Streamlit session memory."""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

def add_to_chat_history(role: str, content: str) -> None:
    """Add a message to chat history."""
    try:
        if "chat_history" in st.session_state:
            st.session_state.chat_history.append({
                "role": role,
                "content": content
            })
    except Exception as e:
        logger.error(f"❌ Error: {e}")

def get_chat_history():
    """Get chat history."""
    return st.session_state.get("chat_history", [])
