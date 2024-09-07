# Text Summarization using Vectors

This notebook demonstrates a simple yet effective approach to text summarization using TF-IDF (Term Frequency-Inverse Document Frequency) vectorization. The method extracts key sentences from a given text based on their importance, as determined by TF-IDF scores.

## Table of Contents
1. [Dependencies](#dependencies)
2. [Data](#data)
3. [Methodology](#methodology)
4. [Key Functions](#key-functions)
5. [Usage](#usage)
6. [Example Output](#example-output)
7. [Limitations and Potential Improvements](#limitations-and-potential-improvements)

## Dependencies

The notebook requires the following Python libraries:
- pandas
- numpy
- textwrap
- nltk
- scikit-learn

Make sure to install these libraries and download the necessary NLTK data:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

## Data

The notebook uses a CSV file named 'bbc_text_cls.csv', which contains news articles with their corresponding labels. The data is loaded into a pandas DataFrame.

## Methodology

The summarization technique follows these steps:

1. Split the text into sentences using NLTK's sentence tokenizer.
2. Create TF-IDF vectors for each sentence using scikit-learn's TfidfVectorizer.
3. Compute a score for each sentence based on the average of its non-zero TF-IDF values.
4. Rank sentences based on their scores.
5. Select top-scoring sentences to form the summary.

## Key Functions

1. `wrap(x)`: A utility function that wraps text for better readability.
2. `get_sentence_score(tfidf_row)`: Computes the score of a sentence based on its TF-IDF vector.
3. `summarize(text)`: The main function that takes a text input and returns a summary.

## Usage

To use the summarizer:

1. Load your text data into the notebook.
2. Call the `summarize(text)` function with your text as the input.
3. The function will print the top 5 sentences along with their scores.

Example:

```python
doc = df[df.labels == 'entertainment']['text'].sample(random_state=123)
summarize(doc.iloc[0].split("\n", 1)[1])
```

## Example Output

The notebook provides two examples of summarization:

1. A business article about Christmas sales.
2. An entertainment article about MTV Music Awards.

For each example, the notebook shows:
- The original full text
- The title of the article
- The generated summary (top 5 sentences with their scores)

## Limitations and Potential Improvements

1. The current implementation always selects the top 5 sentences. This could be improved by making the number of sentences variable based on the input text length or a user-defined parameter.
2. The method doesn't consider the order of sentences in the original text. This could lead to a summary that doesn't flow naturally.
3. The summarizer doesn't handle coreference resolution, which might lead to summaries with unclear pronoun references.
4. The TF-IDF vectorizer is fitted on each text individually, which might not be ideal for capturing broader corpus statistics.
5. The method doesn't account for sentence position, which can be an important factor in some types of texts (e.g., news articles often put key information at the beginning).

Potential improvements could include:
- Implementing a more sophisticated scoring mechanism that considers sentence position, length, and inter-sentence similarity.
- Adding coreference resolution to improve summary coherence.
- Allowing for adjustable summary length based on user input or text characteristics.
- Exploring other text representation methods beyond TF-IDF, such as word embeddings or more recent transformer-based models.

This notebook serves as a good starting point for text summarization and can be extended or modified for more specific use cases or improved performance.
