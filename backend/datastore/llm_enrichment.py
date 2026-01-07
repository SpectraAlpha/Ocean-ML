"""
LLM-based image tag enrichment service
"""
import base64
import logging
from typing import List, Tuple, Optional
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class LLMTagEnricher:
    """Service for enriching image metadata using LLM vision models"""
    
    def __init__(self, api_key: Optional[str] = None, provider: str = "openai"):
        self.api_key = api_key
        self.provider = provider
        self.enabled = api_key is not None
        
        if self.enabled and provider == "openai":
            try:
                import openai
                self.client = openai.OpenAI(api_key=api_key)
            except ImportError:
                logger.warning("OpenAI not installed, LLM tagging disabled")
                self.enabled = False
        elif self.enabled and provider == "anthropic":
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=api_key)
            except ImportError:
                logger.warning("Anthropic not installed, LLM tagging disabled")
                self.enabled = False
    
    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def analyze_image_openai(self, image_path: str) -> Tuple[List[str], str]:
        """Analyze image using OpenAI GPT-4 Vision"""
        if not self.enabled:
            return [], ""
        
        try:
            base64_image = self._encode_image(image_path)
            
            response = self.client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": """Analyze this ocean/marine image for:
                                1. Presence of plastic waste or debris
                                2. Harmful algae blooms (HABs)
                                3. Water quality indicators
                                4. Marine life
                                5. Coastal pollution
                                
                                Provide:
                                - Tags: comma-separated relevant tags
                                - Description: detailed description
                                
                                Format response as JSON:
                                {
                                  "tags": ["tag1", "tag2", ...],
                                  "description": "detailed description"
                                }
                                """
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=500
            )
            
            content = response.choices[0].message.content
            result = json.loads(content)
            return result.get("tags", []), result.get("description", "")
            
        except Exception as e:
            logger.error(f"Error analyzing image with OpenAI: {e}")
            return [], ""
    
    def analyze_image_anthropic(self, image_path: str) -> Tuple[List[str], str]:
        """Analyze image using Anthropic Claude"""
        if not self.enabled:
            return [], ""
        
        try:
            import anthropic
            
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode("utf-8")
            
            message = self.client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=500,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": image_data,
                                },
                            },
                            {
                                "type": "text",
                                "text": """Analyze this ocean/marine image for:
                                1. Presence of plastic waste or debris
                                2. Harmful algae blooms (HABs)
                                3. Water quality indicators
                                4. Marine life
                                5. Coastal pollution
                                
                                Provide JSON response:
                                {
                                  "tags": ["tag1", "tag2", ...],
                                  "description": "detailed description"
                                }
                                """
                            }
                        ],
                    }
                ],
            )
            
            content = message.content[0].text
            result = json.loads(content)
            return result.get("tags", []), result.get("description", "")
            
        except Exception as e:
            logger.error(f"Error analyzing image with Anthropic: {e}")
            return [], ""
    
    def enrich_image(self, image_path: str) -> Tuple[List[str], str]:
        """Enrich image with LLM-generated tags and description"""
        if not self.enabled:
            logger.info("LLM enrichment not enabled")
            return [], ""
        
        if self.provider == "openai":
            return self.analyze_image_openai(image_path)
        elif self.provider == "anthropic":
            return self.analyze_image_anthropic(image_path)
        else:
            logger.warning(f"Unknown provider: {self.provider}")
            return [], ""
    
    def batch_enrich(self, image_paths: List[str]) -> List[Tuple[List[str], str]]:
        """Enrich multiple images"""
        results = []
        for image_path in image_paths:
            tags, description = self.enrich_image(image_path)
            results.append((tags, description))
        return results
