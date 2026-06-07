"""
tests/conftest.py
Global Pytest fixtures to mock heavy weight loaders and API clients.
Prevents unit tests from attempting network calls or loading massive PyTorch models.
"""
import pytest
from unittest.mock import MagicMock, patch
import torch

@pytest.fixture(autouse=True)
def mock_bert_model_loader():
    """Globally mock BERT model and tokenizer from_pretrained methods."""
    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = {
        "input_ids": torch.tensor([[101, 102]]),
        "attention_mask": torch.tensor([[1, 1]])
    }
    
    mock_model = MagicMock()
    mock_model.to.return_value = mock_model
    
    # Mock output logit tensor
    mock_outputs = MagicMock()
    mock_outputs.logits = torch.tensor([[0.0, 1.0, 0.0]]) # Default normal prediction
    mock_model.return_value = mock_outputs

    with patch("transformers.BertTokenizer.from_pretrained", return_value=mock_tokenizer), \
         patch("transformers.BertForSequenceClassification.from_pretrained", return_value=mock_model):
        yield

@pytest.fixture(autouse=True)
def mock_marian_translation_loader():
    """Globally mock Helsinki-NLP MarianMT translation tokenizer and model loaders."""
    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = {
        "input_ids": torch.tensor([[101, 102]]),
        "attention_mask": torch.tensor([[1, 1]])
    }
    
    mock_model = MagicMock()
    # Mock generated token outputs
    mock_model.generate.return_value = torch.tensor([[101, 102, 103]])
    mock_model.to.return_value = mock_model

    with patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer), \
         patch("transformers.AutoModelForSeq2SeqLM.from_pretrained", return_value=mock_model):
        yield

@pytest.fixture(autouse=True)
def mock_groq_client():
    """Globally mock the Groq API client to prevent network access."""
    mock_client = MagicMock()
    
    # Mock chat completions
    mock_completion = MagicMock()
    mock_completion.choices = [MagicMock()]
    mock_completion.choices[0].message.content = "Mocked image analysis response."
    mock_client.chat.completions.create.return_value = mock_completion

    with patch("groq.Groq", return_value=mock_client):
        yield
