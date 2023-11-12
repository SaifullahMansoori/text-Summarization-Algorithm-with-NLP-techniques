from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
import heapq

def frequency_based_summarizer(text, num_sentences=3):
    # Tokenize the text into sentences and words
    sentences = sent_tokenize(text)
    words = word_tokenize(text.lower())  # Convert to lowercase for case-insensitivity

    # Remove stop words
    stop_words = set(stopwords.words("english"))
    words = [word for word in words if word.isalnum() and word not in stop_words]

    # Calculate word frequency
    word_freq = FreqDist(words)

    # Calculate sentence scores based on word frequency
    sentence_scores = {}
    for sentence in sentences:
        for word, freq in word_freq.items():
            if word in sentence.lower():
                if sentence not in sentence_scores:
                    sentence_scores[sentence] = freq
                else:
                    sentence_scores[sentence] += freq

    # Select the top N sentences with highest scores
    summary_sentences = heapq.nlargest(num_sentences, sentence_scores, key=sentence_scores.get)

    # Combine selected sentences to form the summary
    summary = ' '.join(summary_sentences)

    return summary

# Example usage
if __name__ == "__main__":
    # Sample text for summarization
    sample_text = """
    Natural language processing (NLP) is a subfield of artificial intelligence that focuses on the interaction 
    between computers and humans through natural language. It involves the development of algorithms and models 
    that enable computers to understand, interpret, and generate human-like text. NLP has applications in various 
    domains, including machine translation, sentiment analysis, chatbots, and text summarization.

    Text summarization is the process of distilling the most important information from a source text to create a 
    concise and coherent summary. There are different approaches to text summarization, including extractive 
    methods that select and rearrange existing sentences and abstractive methods that generate new sentences 
    to capture the essence of the original text.

    This implementation demonstrates a simple frequency-based text summarization algorithm in Python. The algorithm 
    identifies the most frequent words in the text and extracts sentences containing these words to form the summary.

    To use the frequency-based summarizer, call the function with the text you want to summarize and specify the 
    number of sentences you want in the summary. For example:
    """

    summary = frequency_based_summarizer(sample_text, num_sentences=3)
    print("Original Text:\n", sample_text)
    print("\nSummary:\n", summary)

