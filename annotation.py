import oseti
import os
import csv

analyzer = oseti.Analyzer()

def analyze_text(text):
    """
    Analyzes the given text using OSETI and returns the analysis result.    
    Args:
        text (str): The text to be analyzed.

    Returns:
        dict: The analysis result containing tokens, words, and sentences.
    """
    # Analyze the text
    result = analyzer.analyze(text)

    return result[0]

def analyze_and_save(input_csv, output_csv):
    """
    Reads sentences from a CSV file, performs sentiment analysis, and writes the results to a new CSV file.

    Args:
        input_csv (str): Path to the input CSV file.
        output_csv (str): Path to the output CSV file.
    """
    with open(input_csv, 'r', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        sentences = [row['sentence'] for row in reader]

    results = []
    for sentence in sentences:
        sentiment = analyze_text(sentence)
        # Convert sentiment score to required format
        if sentiment > 0.3:
            sentiment = 0
        elif sentiment < -0.3:
            sentiment = 2
        else:
            sentiment = 1

        # Append the sentence and its sentiment to the results list
        results.append({'sentence': sentence, 'sentiment': sentiment})

    with open(output_csv, 'w', encoding='utf-8', newline='') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=['sentence', 'sentiment'])
        writer.writeheader()
        writer.writerows(results)

# Example usage
input_csv = 'japanese_sentence.csv'
output_csv = 'japanese_sentence_with_sentiment.csv'
analyze_and_save(input_csv, output_csv)