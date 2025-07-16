from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field

class SentimentAnalysisToolInput(BaseModel):
    """Input schema for SentimentAnalysisTool."""
    argument: str = Field(..., description="The text to be analysed for sentiments")

class SentimentAnalysisTool(BaseTool):
    name: str = "Sentiment Analysis Tool"
    description: str = (
        "Analyzes the sentiment of text to ensure positive and engaging communication."
    )
    args_schema: Type[BaseModel] = SentimentAnalysisToolInput

    def _run(self, argument: str) -> str:
        # Your custom code tool goes here
        return "positive"