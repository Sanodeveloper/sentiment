import csv

def split_sentences_to_csv(input_file, output_file):
    """
    Reads a text file, splits its content into sentences, and writes them to a CSV file.

    Args:
        input_file (str): Path to the input text file.
        output_file (str): Path to the output CSV file.
    """
    with open(input_file, 'r', encoding='utf-8') as file:
        text = file.read()

    # Split the text into sentences using '。' as the delimiter
    sentences = [sentence.strip() for sentence in text.split('。') if sentence.strip()]

    # Write the sentences to a CSV file
    with open(output_file, 'w', encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['sentence'])  # Write the header
        for sentence in sentences:
            writer.writerow([sentence])

# Example usage
input_file = 'japanese_sentence.txt'
output_file = 'japanese_sentence.csv'
split_sentences_to_csv(input_file, output_file)