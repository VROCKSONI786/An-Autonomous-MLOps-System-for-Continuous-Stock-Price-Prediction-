from google import genai
from google.genai import types
from dotenv import load_dotenv
import os
import pandas as pd
import logging
import time
import json

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """Analyze sentiment using Gemini LLM"""
    
    def __init__(self):
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        self.client = genai.Client(api_key=api_key)
    
    def analyze_article_sentiment(self, title, summary, symbol=None):
        """Analyze sentiment of a single news article"""
        try:
            # Handle NaN/None values
            title = str(title) if pd.notna(title) else ""
            summary = str(summary) if pd.notna(summary) else ""
            symbol = str(symbol) if pd.notna(symbol) else None
            
            prompt = f"""
            Analyze the sentiment of this financial news article and provide a structured response.
            
            Title: {title}
            Summary: {summary}
            {"Stock: " + symbol if symbol else ""}
            
            Provide your analysis in the following JSON format:
            {{
                "sentiment": "positive/negative/neutral",
                "confidence": 0.0-1.0,
                "impact": "high/medium/low",
                "sector_impact": "sector name or general market",
                "key_themes": ["theme1", "theme2"],
                "reasoning": "brief explanation"
            }}
            
            Only return the JSON, no additional text.
            """
            
            response = self.client.models.generate_content(
                model='gemini-1.5-flash',  # Updated model name
                contents=prompt
            )
            
            # Parse JSON response
            try:
                result = json.loads(response.text)
            except:
                # Fallback if JSON parsing fails
                result = {
                    "sentiment": "neutral",
                    "confidence": 0.5,
                    "impact": "low",
                    "sector_impact": "unknown",
                    "key_themes": [],
                    "reasoning": response.text[:200] if hasattr(response, 'text') else "No response"
                }
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {str(e)}")
            return {
                "sentiment": "neutral",
                "confidence": 0.0,
                "impact": "low",
                "sector_impact": "unknown",
                "key_themes": [],
                "reasoning": f"Error: {str(e)}"
            }
    
    def analyze_bulk_sentiment(self, news_df, delay=2):
        """Analyze sentiment for multiple articles"""
        results = []
        
        for idx, row in news_df.iterrows():
            logger.info(f"Analyzing article {idx+1}/{len(news_df)}: {str(row['title'])[:50]}...")
            
            sentiment = self.analyze_article_sentiment(
                title=row['title'],
                summary=row.get('summary', ''),
                symbol=row.get('symbol', None)
            )
            
            # Combine with original data
            result = {
                **row.to_dict(),
                **sentiment
            }
            results.append(result)
            
            # Rate limiting
            time.sleep(delay)
        
        return pd.DataFrame(results)
    
    def get_market_sentiment_summary(self, sentiment_df):
        """Generate overall market sentiment summary"""
        try:
            # Count sentiments
            sentiment_counts = sentiment_df['sentiment'].value_counts()
            
            # Average confidence
            avg_confidence = sentiment_df['confidence'].mean()
            
            # Sector impacts
            sector_impacts = sentiment_df['sector_impact'].value_counts().head(5)
            
            prompt = f"""
            Based on the following market sentiment data, provide a brief market outlook:
            
            Sentiment Distribution:
            {sentiment_counts.to_dict()}
            
            Average Confidence: {avg_confidence:.2f}
            
            Top Affected Sectors:
            {sector_impacts.to_dict()}
            
            Provide a 3-4 sentence summary of the overall market sentiment and what investors should watch for.
            """
            
            response = self.client.models.generate_content(
                model='gemini-1.5-flash',  # Updated model name
                contents=prompt
            )
            return response.text
            
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return "Unable to generate market summary"

if __name__ == "__main__":
    # Test with sample news
    analyzer = SentimentAnalyzer()
    
    # Test single article
    result = analyzer.analyze_article_sentiment(
        title="Reliance Industries reports strong Q4 earnings",
        summary="Reliance posted a 12% increase in quarterly profits driven by retail and telecom segments",
        symbol="RELIANCE.NS"
    )
    
    print("Sentiment Analysis Result:")
    print(json.dumps(result, indent=2))