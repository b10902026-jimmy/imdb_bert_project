"""
BERT-based sentiment classifier model.

This module defines the BERT-based sentiment classifier model for the IMDB Movie Review dataset.
"""

import logging
from typing import Dict, Any, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import (
    BertModel,
    BertPreTrainedModel,
    BertConfig,
    PreTrainedModel,
    AutoModel,
    AutoConfig,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class BertForSentimentClassification(nn.Module):
    """BERT-based model for sentiment classification."""

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_labels: int = 2,
        dropout_rate: float = 0.1,
        gradient_checkpointing: bool = False,
    ):
        """Initialize the BERT-based sentiment classifier.

        Args:
            model_name: Name of the pre-trained BERT model.
            num_labels: Number of output labels.
            dropout_rate: Dropout rate for the classification head.
            gradient_checkpointing: Whether to use gradient checkpointing to save memory.
        """
        super().__init__()
        
        # Load pre-trained BERT model and configuration
        self.config = AutoConfig.from_pretrained(
            model_name,
            num_labels=num_labels,
        )
        
        self.bert = AutoModel.from_pretrained(
            model_name,
            config=self.config,
        )
        
        # Enable gradient checkpointing if requested
        if gradient_checkpointing:
            self.bert.gradient_checkpointing_enable()
        
        # Classification head
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)
        
        logger.info(f"Initialized BERT-based sentiment classifier with {model_name}")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass of the model.

        Args:
            input_ids: Input token IDs.
            attention_mask: Attention mask.
            token_type_ids: Token type IDs.
            labels: Optional labels for computing the loss.
            output_attentions: Whether to return all attention matrices.

        Returns:
            Dictionary containing the model outputs (logits, loss, attentions).
        """
        # Get BERT outputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=output_attentions,
        )
        
        # Use the [CLS] token representation for classification
        # For models like BERT, last_hidden_state[:, 0] is often preferred over pooler_output for classification
        # sequence_output = outputs.last_hidden_state
        # pooled_output = sequence_output[:, 0] 
        # However, to maintain consistency with typical BertForSequenceClassification, let's use pooler_output
        # If issues arise, consider switching to last_hidden_state[:, 0]
        pooled_output = outputs.pooler_output 
        
        # Apply dropout and classification head
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        # Prepare the output dictionary
        result = {"logits": logits}
        
        # Compute loss if labels are provided
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits.view(-1, self.config.num_labels), labels.view(-1))
            result["loss"] = loss
        
        # Add attentions to output if requested
        if output_attentions:
            result["attentions"] = outputs.attentions
            
        return result


def get_model(
    model_name: str = "bert-base-uncased",
    num_labels: int = 2,
    dropout_rate: float = 0.1,
    gradient_checkpointing: bool = False,
    device: Optional[torch.device] = None,
) -> BertForSentimentClassification:
    """Get a BERT-based sentiment classifier model.

    Args:
        model_name: Name of the pre-trained BERT model.
        num_labels: Number of output labels.
        dropout_rate: Dropout rate for the classification head.
        gradient_checkpointing: Whether to use gradient checkpointing to save memory.
        device: Device to put the model on. If None, will use CUDA if available.

    Returns:
        A BERT-based sentiment classifier model.
    """
    # Determine device if not provided
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model
    model = BertForSentimentClassification(
        model_name=model_name,
        num_labels=num_labels,
        dropout_rate=dropout_rate,
        gradient_checkpointing=gradient_checkpointing,
    )
    
    # Move model to device
    model = model.to(device)
    
    logger.info(f"Model loaded and moved to {device}")
    
    return model


def save_model(
    model: BertForSentimentClassification,
    tokenizer,
    output_dir: str,
    model_name: str = "bert_sentiment_classifier",
) -> None:
    """Save the model and tokenizer.

    Args:
        model: The model to save.
        tokenizer: The tokenizer to save.
        output_dir: Directory to save the model and tokenizer.
        model_name: Name of the model.
    """
    import os
    from pathlib import Path
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    model_path = Path(output_dir) / model_name
    os.makedirs(model_path, exist_ok=True)
    
    logger.info(f"Saving model to {model_path}...")
    
    # Save model state dict
    torch.save(model.state_dict(), model_path / "model.pt")
    
    # Save config
    model.config.save_pretrained(model_path)
    
    # Save tokenizer
    tokenizer.save_pretrained(model_path)
    
    logger.info(f"Model and tokenizer saved to {model_path}")


def load_model(
    model_path: str,
    device: Optional[torch.device] = None,
) -> Tuple[BertForSentimentClassification, Any]:
    """Load a saved model and tokenizer.

    Args:
        model_path: Path to the saved model.
        device: Device to put the model on. If None, will use CUDA if available.

    Returns:
        A tuple containing the loaded model and tokenizer.
    """
    from pathlib import Path
    from transformers import AutoTokenizer
    
    # Determine device if not provided
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load config
    config = AutoConfig.from_pretrained(model_path)
    
    # Initialize model
    model = BertForSentimentClassification(
        model_name=model_path,
        num_labels=config.num_labels,
    )
    
    # Load model state dict
    model_file = Path(model_path) / "model.pt"
    model.load_state_dict(torch.load(model_file, map_location=device))
    
    # Move model to device
    model = model.to(device)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    logger.info(f"Model and tokenizer loaded from {model_path} and moved to {device}")
    
    return model, tokenizer
