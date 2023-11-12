from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize
import heapq

def cosine_similarity_summarizer(text, num_sentences=3):
    # Tokenize the text into sentences
    sentences = sent_tokenize(text)

    # Create a TfidfVectorizer
    vectorizer = TfidfVectorizer(stop_words='english')

    # Fit and transform the sentences to a TF-IDF matrix
    tfidf_matrix = vectorizer.fit_transform(sentences)

    # Calculate cosine similarity between sentences
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Create a dictionary to store sentence scores
    sentence_scores = {sentence: score for sentence, score in zip(sentences, similarity_matrix.sum(axis=1))}

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

    This implementation demonstrates a simple cosine similarity-based text summarization algorithm in Python. The 
    algorithm measures the similarity between sentences using the TF-IDF representation and selects the top N 
    sentences with the highest scores to form the summary.

    To use the cosine similarity summarizer, call the function with the text you want to summarize and specify 
    the number of sentences you want in the summary. For example:
    """

    summary = cosine_similarity_summarizer(sample_text, num_sentences=3)
    print("Original Text:\n", sample_text)
    print("\nSummary:\n", summary)
