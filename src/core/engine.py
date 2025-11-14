"""
Multi-Model Prompt Engineering Platform v2.0
============================================
A production-grade framework for evaluating and comparing multiple LLMs
with advanced bias detection, performance analysis, and quality assessment.

Author: Rosalina Torres
License: MIT
"""

import os
import json
import time
import asyncio
import hashlib
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('llm_evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import LLM libraries
try:
    import openai
    from openai import OpenAI
except ImportError:
    logger.warning("OpenAI library not installed")
    
try:
    from anthropic import Anthropic
except ImportError:
    logger.warning("Anthropic library not installed")
    
try:
    import google.generativeai as genai
except ImportError:
    logger.warning("Google Generative AI library not installed")
    
try:
    import requests  # For Llama/Replicate API
except ImportError:
    logger.warning("Requests library not installed")

from dotenv import load_dotenv
load_dotenv()

# ============================================================================
# Data Classes and Enums
# ============================================================================

class ModelProvider(Enum):
    """Enumeration of supported LLM providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    META = "meta"
    COHERE = "cohere"
    MISTRAL = "mistral"

class EvaluationCategory(Enum):
    """Categories for prompt evaluation"""
    REASONING = "reasoning"
    CREATIVE = "creative"
    TECHNICAL = "technical"
    RESEARCH = "research"
    BUSINESS = "business"
    TRANSLATION = "translation"
    SUMMARIZATION = "summarization"
    CODE_GENERATION = "code_generation"
    CONVERSATION = "conversation"
    FACTUAL = "factual"

class BiasType(Enum):
    """Types of bias to detect"""
    GENDER = "gender"
    RACIAL = "racial"
    POLITICAL = "political"
    CULTURAL = "cultural"
    AGE = "age"
    SOCIOECONOMIC = "socioeconomic"
    RELIGIOUS = "religious"
    CONFIRMATION = "confirmation"

@dataclass
class ModelConfig:
    """Configuration for a specific model"""
    provider: ModelProvider
    model_name: str
    api_key: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 1000
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    timeout: int = 60
    retry_attempts: int = 3
    retry_delay: float = 2.0
    custom_params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PromptTemplate:
    """Template for prompts with metadata"""
    id: str
    content: str
    category: EvaluationCategory
    variables: List[str] = field(default_factory=list)
    expected_format: Optional[str] = None
    evaluation_criteria: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    difficulty: int = 1  # 1-5 scale
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class ModelResponse:
    """Response from an LLM with metadata"""
    model: str
    provider: ModelProvider
    prompt_id: str
    content: str
    response_time: float
    token_count: Optional[int] = None
    cost: Optional[float] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class EvaluationResult:
    """Complete evaluation result for a prompt"""
    prompt_id: str
    category: EvaluationCategory
    responses: List[ModelResponse]
    scores: Dict[str, float] = field(default_factory=dict)
    bias_analysis: Dict[str, Any] = field(default_factory=dict)
    consensus_score: Optional[float] = None
    best_response: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    evaluation_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

# ============================================================================
# Abstract Base Classes
# ============================================================================

class LLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.client = None
        self._initialize_client()
    
    @abstractmethod
    def _initialize_client(self):
        """Initialize the provider-specific client"""
        pass
    
    @abstractmethod
    async def generate_async(self, prompt: str) -> ModelResponse:
        """Generate response asynchronously"""
        pass
    
    def generate(self, prompt: str, prompt_id: str = "") -> ModelResponse:
        """Generate response synchronously"""
        return asyncio.run(self.generate_async(prompt, prompt_id))
    
    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost based on token usage"""
        # Override in subclasses with provider-specific pricing
        return 0.0

class BiasDetector(ABC):
    """Abstract base class for bias detection"""
    
    @abstractmethod
    def detect(self, text: str) -> Dict[str, float]:
        """Detect bias in text and return scores"""
        pass

class QualityEvaluator(ABC):
    """Abstract base class for quality evaluation"""
    
    @abstractmethod
    def evaluate(self, response: str, criteria: List[str]) -> Dict[str, float]:
        """Evaluate response quality based on criteria"""
        pass

# ============================================================================
# LLM Provider Implementations
# ============================================================================

class OpenAIProvider(LLMProvider):
    """OpenAI GPT provider implementation"""
    
    def _initialize_client(self):
        api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not provided")
        self.client = OpenAI(api_key=api_key)
    
    async def generate_async(self, prompt: str, prompt_id: str = "") -> ModelResponse:
        start_time = time.time()
        try:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.config.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                top_p=self.config.top_p,
                frequency_penalty=self.config.frequency_penalty,
                presence_penalty=self.config.presence_penalty,
                **self.config.custom_params
            )
            
            content = response.choices[0].message.content
            response_time = time.time() - start_time
            
            # Calculate tokens and cost
            total_tokens = response.usage.total_tokens if response.usage else None
            cost = self._calculate_cost(
                response.usage.prompt_tokens if response.usage else 0,
                response.usage.completion_tokens if response.usage else 0
            )
            
            return ModelResponse(
                model=self.config.model_name,
                provider=ModelProvider.OPENAI,
                prompt_id=prompt_id,
                content=content,
                response_time=response_time,
                token_count=total_tokens,
                cost=cost,
                metadata={"finish_reason": response.choices[0].finish_reason}
            )
        except Exception as e:
            logger.error(f"OpenAI generation error: {str(e)}")
            return ModelResponse(
                model=self.config.model_name,
                provider=ModelProvider.OPENAI,
                prompt_id=prompt_id,
                content="",
                response_time=time.time() - start_time,
                error=str(e)
            )
    
    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate OpenAI API cost"""
        pricing = {
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
            "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015}
        }
        
        model_pricing = pricing.get(self.config.model_name, {"input": 0, "output": 0})
        return (input_tokens * model_pricing["input"] + 
                output_tokens * model_pricing["output"]) / 1000

class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider implementation"""
    
    def _initialize_client(self):
        api_key = self.config.api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Anthropic API key not provided")
        self.client = Anthropic(api_key=api_key)
    
    async def generate_async(self, prompt: str, prompt_id: str = "") -> ModelResponse:
        start_time = time.time()
        try:
            response = await asyncio.to_thread(
                self.client.messages.create,
                model=self.config.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                **self.config.custom_params
            )
            
            content = response.content[0].text if response.content else ""
            response_time = time.time() - start_time
            
            return ModelResponse(
                model=self.config.model_name,
                provider=ModelProvider.ANTHROPIC,
                prompt_id=prompt_id,
                content=content,
                response_time=response_time,
                metadata={"stop_reason": response.stop_reason if hasattr(response, 'stop_reason') else None}
            )
        except Exception as e:
            logger.error(f"Anthropic generation error: {str(e)}")
            return ModelResponse(
                model=self.config.model_name,
                provider=ModelProvider.ANTHROPIC,
                prompt_id=prompt_id,
                content="",
                response_time=time.time() - start_time,
                error=str(e)
            )

class GoogleProvider(LLMProvider):
    """Google Gemini provider implementation"""
    
    def _initialize_client(self):
        api_key = self.config.api_key or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Google API key not provided")
        genai.configure(api_key=api_key)
        self.client = genai.GenerativeModel(self.config.model_name)
    
    async def generate_async(self, prompt: str, prompt_id: str = "") -> ModelResponse:
        start_time = time.time()
        try:
            generation_config = genai.types.GenerationConfig(
                temperature=self.config.temperature,
                max_output_tokens=self.config.max_tokens,
                top_p=self.config.top_p
            )
            
            response = await asyncio.to_thread(
                self.client.generate_content,
                prompt,
                generation_config=generation_config
            )
            
            content = response.text
            response_time = time.time() - start_time
            
            return ModelResponse(
                model=self.config.model_name,
                provider=ModelProvider.GOOGLE,
                prompt_id=prompt_id,
                content=content,
                response_time=response_time
            )
        except Exception as e:
            logger.error(f"Google generation error: {str(e)}")
            return ModelResponse(
                model=self.config.model_name,
                provider=ModelProvider.GOOGLE,
                prompt_id=prompt_id,
                content="",
                response_time=time.time() - start_time,
                error=str(e)
            )

class MetaLlamaProvider(LLMProvider):
    """Meta Llama provider implementation (via Replicate or local)"""
    
    def _initialize_client(self):
        self.api_key = self.config.api_key or os.getenv("REPLICATE_API_KEY")
        if not self.api_key:
            raise ValueError("Replicate API key not provided for Llama")
        self.api_url = "https://api.replicate.com/v1/predictions"
    
    async def generate_async(self, prompt: str, prompt_id: str = "") -> ModelResponse:
        start_time = time.time()
        try:
            headers = {
                "Authorization": f"Token {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "version": self.config.custom_params.get("version", "latest"),
                "input": {
                    "prompt": prompt,
                    "temperature": self.config.temperature,
                    "max_tokens": self.config.max_tokens,
                    "top_p": self.config.top_p
                }
            }
            
            response = await asyncio.to_thread(
                requests.post,
                self.api_url,
                headers=headers,
                json=data,
                timeout=self.config.timeout
            )
            
            response.raise_for_status()
            result = response.json()
            
            # Poll for completion
            prediction_url = result.get("urls", {}).get("get")
            if prediction_url:
                content = await self._poll_for_completion(prediction_url, headers)
            else:
                content = result.get("output", "")
            
            response_time = time.time() - start_time
            
            return ModelResponse(
                model=self.config.model_name,
                provider=ModelProvider.META,
                prompt_id=prompt_id,
                content=content,
                response_time=response_time
            )
        except Exception as e:
            logger.error(f"Llama generation error: {str(e)}")
            return ModelResponse(
                model=self.config.model_name,
                provider=ModelProvider.META,
                prompt_id=prompt_id,
                content="",
                response_time=time.time() - start_time,
                error=str(e)
            )
    
    async def _poll_for_completion(self, url: str, headers: dict) -> str:
        """Poll Replicate API for completion"""
        max_attempts = 60
        for _ in range(max_attempts):
            response = await asyncio.to_thread(requests.get, url, headers=headers)
            result = response.json()
            
            if result.get("status") == "succeeded":
                return "".join(result.get("output", []))
            elif result.get("status") == "failed":
                raise Exception(f"Prediction failed: {result.get('error')}")
            
            await asyncio.sleep(1)
        
        raise Exception("Prediction timed out")

# ============================================================================
# Evaluation Components
# ============================================================================

class AdvancedBiasDetector(BiasDetector):
    """Advanced bias detection using multiple techniques"""
    
    def __init__(self):
        self.bias_keywords = self._load_bias_keywords()
        self.bias_patterns = self._load_bias_patterns()
    
    def _load_bias_keywords(self) -> Dict[BiasType, List[str]]:
        """Load bias-indicative keywords"""
        return {
            BiasType.GENDER: [
                "he", "she", "man", "woman", "male", "female", 
                "masculine", "feminine", "gender"
            ],
            BiasType.RACIAL: [
                "race", "ethnic", "white", "black", "asian", 
                "hispanic", "latino", "african"
            ],
            BiasType.POLITICAL: [
                "liberal", "conservative", "democrat", "republican",
                "left-wing", "right-wing", "socialist", "capitalist"
            ],
            BiasType.CULTURAL: [
                "western", "eastern", "american", "european", 
                "asian", "african", "culture", "tradition"
            ],
            BiasType.AGE: [
                "young", "old", "elderly", "youth", "millennial",
                "boomer", "generation", "age"
            ],
            BiasType.SOCIOECONOMIC: [
                "rich", "poor", "wealthy", "poverty", "class",
                "income", "education", "privilege"
            ],
            BiasType.RELIGIOUS: [
                "christian", "muslim", "jewish", "hindu", "buddhist",
                "atheist", "religious", "secular", "faith"
            ]
        }
    
    def _load_bias_patterns(self) -> Dict[BiasType, List[str]]:
        """Load regex patterns for bias detection"""
        return {
            BiasType.GENDER: [
                r"\b(he|she)\s+(always|never|typically|usually)\b",
                r"\b(men|women)\s+(are|tend to be)\s+\w+er\b"
            ],
            BiasType.CONFIRMATION: [
                r"\b(obviously|clearly|everyone knows|it's common sense)\b",
                r"\b(always|never|all|none|every|no one)\b"
            ]
        }
    
    def detect(self, text: str) -> Dict[str, float]:
        """Detect various types of bias in text"""
        text_lower = text.lower()
        results = {}
        
        for bias_type in BiasType:
            score = 0.0
            
            # Keyword frequency analysis
            if bias_type in self.bias_keywords:
                keywords = self.bias_keywords[bias_type]
                keyword_count = sum(1 for keyword in keywords if keyword in text_lower)
                score += min(keyword_count / len(keywords), 1.0) * 0.5
            
            # Pattern matching
            if bias_type in self.bias_patterns:
                import re
                patterns = self.bias_patterns[bias_type]
                pattern_matches = sum(
                    len(re.findall(pattern, text_lower, re.IGNORECASE))
                    for pattern in patterns
                )
                score += min(pattern_matches / 10, 1.0) * 0.5
            
            results[bias_type.value] = round(score, 3)
        
        # Calculate overall bias score
        results["overall"] = round(
            sum(results.values()) / len(results) if results else 0,
            3
        )
        
        return results

class ComprehensiveQualityEvaluator(QualityEvaluator):
    """Comprehensive quality evaluation system"""
    
    def __init__(self):
        self.criteria_weights = {
            "relevance": 0.25,
            "coherence": 0.20,
            "completeness": 0.20,
            "accuracy": 0.15,
            "clarity": 0.10,
            "creativity": 0.10
        }
    
    def evaluate(self, response: str, criteria: List[str]) -> Dict[str, float]:
        """Evaluate response quality across multiple dimensions"""
        scores = {}
        
        # Basic text analysis
        word_count = len(response.split())
        sentence_count = len(response.split('.'))
        avg_sentence_length = word_count / max(sentence_count, 1)
        
        # Relevance score (simplified - would use embeddings in production)
        scores["relevance"] = min(word_count / 100, 1.0)
        
        # Coherence score (based on sentence structure)
        scores["coherence"] = min(avg_sentence_length / 20, 1.0)
        
        # Completeness score
        scores["completeness"] = min(word_count / 200, 1.0)
        
        # Accuracy score (placeholder - would use fact-checking in production)
        scores["accuracy"] = 0.8  # Default reasonable accuracy
        
        # Clarity score (based on sentence complexity)
        scores["clarity"] = 1.0 - min(avg_sentence_length / 50, 0.5)
        
        # Creativity score (based on vocabulary diversity)
        unique_words = len(set(response.lower().split()))
        scores["creativity"] = min(unique_words / word_count, 1.0) if word_count > 0 else 0
        
        # Calculate weighted overall score
        overall_score = sum(
            scores.get(criterion, 0) * self.criteria_weights.get(criterion, 0)
            for criterion in self.criteria_weights
        )
        scores["overall"] = round(overall_score, 3)
        
        # Apply custom criteria if provided
        for criterion in criteria:
            if criterion not in scores:
                scores[criterion] = self._evaluate_custom_criterion(response, criterion)
        
        return {k: round(v, 3) for k, v in scores.items()}
    
    def _evaluate_custom_criterion(self, response: str, criterion: str) -> float:
        """Evaluate custom criterion"""
        # Simplified evaluation - override for specific criteria
        return 0.7

# ============================================================================
# Main Evaluation Engine
# ============================================================================

class MultiModelEvaluationEngine:
    """Main engine for multi-model prompt evaluation"""
    
    def __init__(self, configs: List[ModelConfig] = None):
        self.providers = {}
        self.bias_detector = AdvancedBiasDetector()
        self.quality_evaluator = ComprehensiveQualityEvaluator()
        self.results_cache = {}
        self.evaluation_history = []
        
        if configs:
            self.initialize_providers(configs)
    
    def initialize_providers(self, configs: List[ModelConfig]):
        """Initialize LLM providers"""
        for config in configs:
            try:
                if config.provider == ModelProvider.OPENAI:
                    self.providers[config.model_name] = OpenAIProvider(config)
                elif config.provider == ModelProvider.ANTHROPIC:
                    self.providers[config.model_name] = AnthropicProvider(config)
                elif config.provider == ModelProvider.GOOGLE:
                    self.providers[config.model_name] = GoogleProvider(config)
                elif config.provider == ModelProvider.META:
                    self.providers[config.model_name] = MetaLlamaProvider(config)
                logger.info(f"Initialized provider: {config.model_name}")
            except Exception as e:
                logger.error(f"Failed to initialize {config.model_name}: {str(e)}")
    
    async def evaluate_prompt_async(
        self,
        prompt_template: PromptTemplate,
        variables: Dict[str, str] = None
    ) -> EvaluationResult:
        """Evaluate a prompt across all models asynchronously"""
        start_time = time.time()
        
        # Format prompt with variables
        prompt = self._format_prompt(prompt_template, variables)
        
        # Check cache
        cache_key = self._generate_cache_key(prompt)
        if cache_key in self.results_cache:
            logger.info(f"Returning cached result for prompt {prompt_template.id}")
            return self.results_cache[cache_key]
        
        # Generate responses from all models
        tasks = []
        for model_name, provider in self.providers.items():
            task = provider.generate_async(prompt, prompt_template.id)
            tasks.append(task)
        
        responses = await asyncio.gather(*tasks)
        
        # Filter out error responses
        valid_responses = [r for r in responses if not r.error]
        
        # Perform quality evaluation
        scores = {}
        for response in valid_responses:
            quality_scores = self.quality_evaluator.evaluate(
                response.content,
                prompt_template.evaluation_criteria
            )
            scores[response.model] = quality_scores
        
        # Perform bias analysis
        bias_analysis = {}
        for response in valid_responses:
            bias_scores = self.bias_detector.detect(response.content)
            bias_analysis[response.model] = bias_scores
        
        # Calculate consensus score
        consensus_score = self._calculate_consensus(valid_responses)
        
        # Determine best response
        best_response = self._select_best_response(valid_responses, scores)
        
        # Create evaluation result
        result = EvaluationResult(
            prompt_id=prompt_template.id,
            category=prompt_template.category,
            responses=responses,
            scores=scores,
            bias_analysis=bias_analysis,
            consensus_score=consensus_score,
            best_response=best_response,
            evaluation_time=time.time() - start_time,
            metadata={
                "prompt_difficulty": prompt_template.difficulty,
                "valid_responses": len(valid_responses),
                "total_responses": len(responses)
            }
        )
        
        # Cache and store result
        self.results_cache[cache_key] = result
        self.evaluation_history.append(result)
        
        return result
    
    def evaluate_prompt(
        self,
        prompt_template: PromptTemplate,
        variables: Dict[str, str] = None
    ) -> EvaluationResult:
        """Evaluate a prompt synchronously"""
        return asyncio.run(self.evaluate_prompt_async(prompt_template, variables))
    
    def batch_evaluate(
        self,
        prompts: List[PromptTemplate],
        parallel: bool = True,
        max_workers: int = 5
    ) -> List[EvaluationResult]:
        """Evaluate multiple prompts in batch"""
        results = []
        
        if parallel:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(self.evaluate_prompt, prompt)
                    for prompt in prompts
                ]
                
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Batch evaluation error: {str(e)}")
        else:
            for prompt in prompts:
                try:
                    result = self.evaluate_prompt(prompt)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Evaluation error for prompt {prompt.id}: {str(e)}")
        
        return results
    
    def _format_prompt(
        self,
        template: PromptTemplate,
        variables: Dict[str, str] = None
    ) -> str:
        """Format prompt with variables"""
        prompt = template.content
        if variables:
            for key, value in variables.items():
                prompt = prompt.replace(f"{{{key}}}", value)
        return prompt
    
    def _generate_cache_key(self, prompt: str) -> str:
        """Generate cache key for prompt"""
        return hashlib.md5(prompt.encode()).hexdigest()
    
    def _calculate_consensus(self, responses: List[ModelResponse]) -> float:
        """Calculate consensus score among responses"""
        if len(responses) < 2:
            return 1.0
        
        # Simplified consensus using response length similarity
        lengths = [len(r.content) for r in responses]
        mean_length = statistics.mean(lengths)
        std_dev = statistics.stdev(lengths) if len(lengths) > 1 else 0
        
        # Normalize to 0-1 scale (lower std dev = higher consensus)
        consensus = 1.0 - min(std_dev / mean_length, 1.0) if mean_length > 0 else 0
        
        return round(consensus, 3)
    
    def _select_best_response(
        self,
        responses: List[ModelResponse],
        scores: Dict[str, Dict[str, float]]
    ) -> str:
        """Select best response based on scores"""
        if not responses:
            return None
        
        best_model = max(
            scores.keys(),
            key=lambda m: scores[m].get("overall", 0)
        ) if scores else responses[0].model
        
        for response in responses:
            if response.model == best_model:
                return response.model
        
        return responses[0].model

    def generate_report(
        self,
        results: List[EvaluationResult],
        output_format: str = "json"
    ) -> Union[str, pd.DataFrame]:
        """Generate comprehensive evaluation report"""
        if output_format == "json":
            return json.dumps(
                [asdict(r) for r in results],
                default=str,
                indent=2
            )
        elif output_format == "dataframe":
            data = []
            for result in results:
                for response in result.responses:
                    data.append({
                        "prompt_id": result.prompt_id,
                        "category": result.category.value,
                        "model": response.model,
                        "provider": response.provider.value,
                        "response_time": response.response_time,
                        "error": response.error,
                        "quality_score": result.scores.get(
                            response.model, {}
                        ).get("overall", 0),
                        "bias_score": result.bias_analysis.get(
                            response.model, {}
                        ).get("overall", 0),
                        "consensus_score": result.consensus_score,
                        "is_best": result.best_response == response.model
                    })
            return pd.DataFrame(data)
        elif output_format == "markdown":
            return self._generate_markdown_report(results)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
    
    def _generate_markdown_report(self, results: List[EvaluationResult]) -> str:
        """Generate markdown report"""
        report = ["# Multi-Model Evaluation Report\n"]
        report.append(f"Generated: {datetime.now().isoformat()}\n")
        report.append(f"Total Evaluations: {len(results)}\n\n")
        
        for result in results:
            report.append(f"## Prompt: {result.prompt_id}\n")
            report.append(f"**Category:** {result.category.value}\n")
            report.append(f"**Evaluation Time:** {result.evaluation_time:.2f}s\n")
            report.append(f"**Consensus Score:** {result.consensus_score}\n")
            report.append(f"**Best Response:** {result.best_response}\n\n")
            
            report.append("### Model Performance\n")
            report.append("| Model | Quality | Bias | Response Time |\n")
            report.append("|-------|---------|------|---------------|\n")
            
            for response in result.responses:
                quality = result.scores.get(response.model, {}).get("overall", "N/A")
                bias = result.bias_analysis.get(response.model, {}).get("overall", "N/A")
                report.append(
                    f"| {response.model} | {quality} | {bias} | "
                    f"{response.response_time:.2f}s |\n"
                )
            
            report.append("\n")
        
        return "".join(report)

# ============================================================================
# Utility Functions
# ============================================================================

def load_prompts_from_file(filepath: str) -> List[PromptTemplate]:
    """Load prompts from JSON file"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    prompts = []
    for item in data:
        prompts.append(PromptTemplate(
            id=item.get("id"),
            content=item.get("content"),
            category=EvaluationCategory[item.get("category", "REASONING").upper()],
            variables=item.get("variables", []),
            expected_format=item.get("expected_format"),
            evaluation_criteria=item.get("evaluation_criteria", []),
            tags=item.get("tags", []),
            difficulty=item.get("difficulty", 1)
        ))
    
    return prompts

def save_results_to_file(results: List[EvaluationResult], filepath: str):
    """Save evaluation results to file"""
    with open(filepath, 'w') as f:
        json.dump(
            [asdict(r) for r in results],
            f,
            default=str,
            indent=2
        )

# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main execution function"""
    # Configure models
    configs = [
        ModelConfig(
            provider=ModelProvider.OPENAI,
            model_name="gpt-4",
            temperature=0.7
        ),
        ModelConfig(
            provider=ModelProvider.ANTHROPIC,
            model_name="claude-3-sonnet-20240229",
            temperature=0.7
        ),
        ModelConfig(
            provider=ModelProvider.GOOGLE,
            model_name="gemini-pro",
            temperature=0.7
        ),
        ModelConfig(
            provider=ModelProvider.META,
            model_name="llama-2-70b",
            temperature=0.7
        )
    ]
    
    # Initialize engine
    engine = MultiModelEvaluationEngine(configs)
    
    # Create sample prompts
    sample_prompts = [
        PromptTemplate(
            id="reasoning_001",
            content="Solve this logic puzzle: If all roses are flowers, and some flowers fade quickly, can we conclude that some roses fade quickly? Explain your reasoning step by step.",
            category=EvaluationCategory.REASONING,
            evaluation_criteria=["relevance", "coherence", "accuracy"],
            difficulty=3
        ),
        PromptTemplate(
            id="creative_001",
            content="Write a short story (200 words) about a time traveler who can only move forward in time by exactly 7 days at a time.",
            category=EvaluationCategory.CREATIVE,
            evaluation_criteria=["creativity", "coherence", "completeness"],
            difficulty=4
        ),
        PromptTemplate(
            id="technical_001",
            content="Explain the concept of recursion in programming and provide an example of when it would be more appropriate than iteration.",
            category=EvaluationCategory.TECHNICAL,
            evaluation_criteria=["accuracy", "clarity", "completeness"],
            difficulty=2
        )
    ]
    
    # Evaluate prompts
    logger.info("Starting multi-model evaluation...")
    results = engine.batch_evaluate(sample_prompts, parallel=True)
    
    # Generate reports
    json_report = engine.generate_report(results, "json")
    df_report = engine.generate_report(results, "dataframe")
    md_report = engine.generate_report(results, "markdown")
    
    # Save results
    save_results_to_file(results, "evaluation_results.json")
    
    # Save markdown report
    with open("evaluation_report.md", "w") as f:
        f.write(md_report)
    
    # Display summary statistics
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(df_report.groupby("model").agg({
        "quality_score": "mean",
        "bias_score": "mean",
        "response_time": "mean",
        "error": lambda x: (x.notna()).sum()
    }).round(3))
    
    logger.info("Evaluation complete!")

if __name__ == "__main__":
    main()
