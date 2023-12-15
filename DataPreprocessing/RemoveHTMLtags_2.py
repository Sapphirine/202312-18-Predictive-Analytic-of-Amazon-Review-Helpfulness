import re


# Define a function to remove '<br />' and '<p>' tags from a string
def remove_tags(text):
    # Define regular expressions to match '<br />' and '<p>' tags
    br_pattern = re.compile(r'<br\s*/?>', re.IGNORECASE)
    p_pattern = re.compile(r'<p\s*/?>', re.IGNORECASE)
    pattern = re.compile(r'<.*?>', re.DOTALL)
    # Remove the tags from the text
    text = br_pattern.sub('', text)
    text = p_pattern.sub('', text)
    text = pattern.sub('', text)
    return text


if __name__ == '__main__':
    # Read the content of 'output.txt'
    with open('output5.txt', 'r', encoding='utf-8') as input_file:
        content = input_file.read()

    # Remove '<br />' and '<p>' tags from the content
    content_without_tags = remove_tags(content)

    # Write the modified content back to 'output.txt'
    with open('output5.txt', 'w', encoding='utf-8') as output_file:
        output_file.write(content_without_tags)
