# Employee Sentiment Analysis functionality using Gemini AI
import os 
import json
import re
import nltk
from google import genai
from dotenv import load_dotenv

def download_nltk_resources():
    """Ensure required NLTK resources are available"""
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
    except LookupError:
        print("Downloading NLTK resources...")
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')

class EmployeeSentimentAnalyzer:
    def __init__(self):
        load_dotenv()
        self.api_key =  os.getenv("GEMINI_API_KEY")

        if not self.api_key:
            raise ValueError("Gemini API key not provided. Set GEMINI_API_KEY in .env or pass it directly.")
        # Ensure NLTK resources are downloaded
        download_nltk_resources()
        
        
        # Initialize Gemini client
        self.client = genai.Client(api_key=self.api_key)
        
        # Define topics for categorization
        self.topics = [
            "work-life balance", 
            "compensation", 
            "management", 
            "career growth", 
            "company culture", 
            "job satisfaction",
            "workload",
            "remote work",
            "benefits"
        ]
    def gemini_feedback(self, feedback_text: str):
        """
        Analyze employee feedback to predict attrition risk
        and recommend engagement strategies using Gemini.
        """
        if not feedback_text.strip():
            return {
                "error": "Feedback text is empty",
                "sentiment": "neutral",
                "risk_score": 0,
                "risk_level": "Unknown",
                "recommendations": []
            }
        # Create prompt for Gemini
        prompt = f"""
        You are an expert HR AI assistant.
        Analyze the following employee feedback to predict attrition risk and suggest engagement strategies.

        Employee Feedback:
        {feedback_text}

        Instructions:
        1. Analyze the sentiment (positive, negative, neutral).
        2. Predict the attrition risk score (0-100).
        3. Classify the risk level as Low, Medium, or High:
           - 0-40 = Low
           - 41-70 = Medium
           - 71-100 = High
        4. Generate a list of 3-5 actionable recommendations to improve employee engagement.

        Return only valid JSON with this exact structure:
        {{
            "sentiment": "positive | negative | neutral",
            "risk_score": 0-100,
            "risk_level": "Low | Medium | High",
            "recommendations": [
                "Recommendation 1",
                "Recommendation 2"
            ]
        }}
        """

        # Call Gemini
        response = self.client.models.generate_content(
            model="gemini-1.5-flash-latest",
            contents=prompt
        )

        raw_output = response.text.strip()

        # Clean up Gemini output by removing code fences if present
        cleaned_output = re.sub(r"```[a-zA-Z]*\n?", "", raw_output).replace("```", "").strip()

        try:
            # Parse the JSON safely
            result = json.loads(cleaned_output)
        except json.JSONDecodeError:
            # Fallback if model output isn't valid JSON
            return {
                "error": "Failed to parse response",
                "raw_output": raw_output
            }

        return result
    
    
    
   
    
    
    
    
    def analyze_feedback(self, feedback_text: str, past_feedback=None):
        """Main function to analyze employee feedback"""
        if not feedback_text.strip():
            return {
                "error": "No feedback provided",
                "sentiment": {"sentiment": "neutral", "score": 0.5},
                "topics": {"topics": [], "scores": []},
                "attrition_risk": {"risk_score": 0.0, "risk_level": "Unknown"},
                "recommendations": ["Cannot analyze - no feedback provided"]
            }
        
        sentiment = self.gemini_feedback(feedback_text)
        sentiment_score = sentiment.get("risk_score", 0)
        recommendations = sentiment.get("recommendations", [])
       
       
        
        return {
            "sentiment": sentiment,
            "risk_score": sentiment_score,
            "recommendations": recommendations
           
        }

    def _safe_parse(self, response_text, fallback):
        """Try to parse Gemini JSON response safely"""
        import json
        try:
            return json.loads(response_text)
        except Exception:
            return fallback
