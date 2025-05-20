# BERT Sentiment Analysis Model Explanation System

This document describes the model explanation system that has been implemented for the IMDB BERT sentiment analysis model. The system provides users with insights into how the model makes its predictions.

## Overview

The explanation system provides the following types of insights:

1. **Key Influential Words**: Words that most influenced the model's prediction, with their relative importance scores.
2. **Top Phrases**: Phrases that had significant impact on the prediction.
3. **Token-level Heatmap**: Visual representation of each token's importance.
4. **Attention Matrices**: Visualizations of attention patterns across different layers and heads.
5. **Layer Contributions**: Analysis of how much each transformer layer contributed to the final prediction.

## How to Use

### Command Line Usage

You can generate explanations when making predictions using these command-line flags:

```bash
# Basic prediction with explanation
python src/predict.py --model-path models/test_checkpoint \
    --text "This movie was absolutely brilliant!" \
    --explain

# HTML report format (default)
python src/predict.py --model-path models/test_checkpoint \
    --text "This movie was terrible!" \
    --explain --explanation-format html

# CLI text format
python src/predict.py --model-path models/test_checkpoint \
    --text "This movie was amazing!" \
    --explain --explanation-format cli
```

### Viewing Explanations

After running the prediction with explanations:

1. **CLI Output**: If using CLI format, explanations will be printed directly to the console.
2. **HTML Report**: If using HTML format (default), an HTML report will be generated in the output directory.
3. **JSON Results**: All explanations are also included in the prediction JSON result file.

You can also use the `src/display_predictions.py` script to view the prediction results:

```bash
python src/display_predictions.py predictions/prediction_result.json
```

### Programmatic Usage

You can use the explanation system programmatically:

```python
from src.utils.explanation_manager import ExplanationManager
from src.models.bert_classifier import load_model

# Load model and tokenizer
model, tokenizer = load_model("models/test_checkpoint", device)

# Create explanation manager
explanation_manager = ExplanationManager(model, tokenizer, device)

# Generate explanations
explanation_results = explanation_manager.explain_prediction(
    "This movie was fantastic!",
    output_dir="predictions/explanations",
    top_k_words=10,
    max_attention_heads=3,
    layer_contributions=True
)

# Generate HTML report
report_path = explanation_manager.generate_html_report(
    explanation_results,
    "predictions/explanations",
    filename="explanation_report"
)
```

## Explanation Types

### Key Influential Words

Words that had the most impact on the model's prediction, based on attention weights from the classification token. These words are shown with their relative importance as percentages.

### Top Phrases

Short phrases extracted from the input text that were influential in the model's decision. These are identified by analyzing token saliency in context.

### Token-level Heatmap

A visual heatmap showing the importance of each token in the input text. This helps to identify which parts of the text were most salient for the prediction.

### Attention Matrices

Visualizations of the model's attention patterns, showing how different parts of the text are connected in the model's processing. These are available for different layers and attention heads.

### Layer Contributions

Analysis of how much each transformer layer contributed to the final prediction, helping to understand which layers were most important for this specific classification task.

## Example Output

The HTML report provides a comprehensive visualization that includes:

- The input text and prediction result
- Color-coded key words with their importance scores
- Top influential phrases
- Token saliency heatmap
- Interactive attention matrix visualizations
- Layer contribution bar charts

The CLI output provides a condensed version with:

- Prediction and confidence score
- Key influential words with scores
- Top phrases
- Most important layer contributions

## Implementation Details

The explanation system is implemented in:

- `src/utils/explanation_manager.py`: Core implementation of the explanation system
- `src/predict.py`: Integration with the prediction pipeline
- `src/display_predictions.py`: Enhanced display of prediction results with explanations

The system leverages attention weights from the BERT model to generate insights into the model's decision-making process, providing users with a clearer understanding of how their text was classified.
