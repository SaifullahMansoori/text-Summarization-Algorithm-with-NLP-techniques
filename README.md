# Text Summarization Algorithms

This repository contains Python implementations of three text summarization algorithms using Natural Language Processing (NLP) techniques.

## Algorithms

1. **Frequency-based Summarization:**
   - This algorithm summarizes text by identifying and extracting the most frequently occurring words or phrases.

2. **Luhn Algorithm:**
   - The Luhn algorithm ranks sentences based on the presence of important keywords, providing a way to extract key sentences for summarization.

3. **Cosine Similarity Algorithm:**
   - This algorithm uses cosine similarity to measure the similarity between sentences, ranking them to extract the most representative sentences for summarization.

## Usage

### Prerequisites

- Python 3.x
- Install required packages: `pip install -r requirements.txt`

### Example Usage

```python
# Import the summarization algorithms
from frequency_based_summarizer import FrequencyBasedSummarizer
from luhn_algorithm import LuhnAlgorithmSummarizer
from cosine_similarity_summarizer import CosineSimilaritySummarizer

# Sample text for summarization
text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. ..."

# Frequency-based summarization
freq_summarizer = FrequencyBasedSummarizer()
freq_summary = freq_summarizer.summarize(text)
print("Frequency-based Summary:", freq_summary)

# Luhn algorithm summarization
luhn_summarizer = LuhnAlgorithmSummarizer()
luhn_summary = luhn_summarizer.summarize(text)
print("Luhn Algorithm Summary:", luhn_summary)

# Cosine similarity summarization
cosine_summarizer = CosineSimilaritySummarizer()
cosine_summary = cosine_summarizer.summarize(text)
print("Cosine Similarity Summary:", cosine_summary)
