"""Pytest configuration and fixtures"""
import pytest
import asyncio
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def sample_prompt():
    """Sample prompt for testing"""
    from core.models import PromptTemplate, EvaluationCategory
    return PromptTemplate(
        id="test_001",
        content="Test prompt content",
        category=EvaluationCategory.REASONING,
        evaluation_criteria=["accuracy", "coherence"],
        difficulty=3
    )

@pytest.fixture
def mock_model_config():
    """Mock model configuration"""
    from core.models import ModelConfig, ModelProvider
    return ModelConfig(
        provider=ModelProvider.OPENAI,
        model_name="gpt-3.5-turbo",
        temperature=0.7,
        max_tokens=100
    )
