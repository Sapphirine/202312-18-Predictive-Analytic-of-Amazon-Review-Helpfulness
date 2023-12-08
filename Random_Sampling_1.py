def extract_and_save_first_percent(input_filename, output_filename):
    with open(input_filename, 'rb') as input_file:
        content_bytes = input_file.read()

    try:
        # Attempt to decode the content using 'utf-8' encoding
        content = content_bytes.decode('utf-8')
    except UnicodeDecodeError:
        # If 'utf-8' decoding fails, try decoding with 'latin-1' encoding
        content = content_bytes.decode('latin-1', errors='ignore')

    # Calculate the length of the file content
    total_length = len(content)

    # Calculate the number of characters to extract (1% of the total length)
    num_chars_to_extract = total_length // 20

    # Extract the first 1% of the content
    extracted_content = content[:num_chars_to_extract]

    # Write the extracted content to the output file
    with open(output_filename, 'w', encoding='utf-8') as output_file:
        output_file.write(extracted_content)


if __name__ == '__main__':
    input_filename = 'C:/Users/Lenovo/Desktop/MSEE/EECS E6893 Big Data Analytics/Project/data/movies.txt'  # Replace with your input file name
    output_filename = 'output5.txt'  # Replace with the desired output file name
    extract_and_save_first_percent(input_filename, output_filename)
