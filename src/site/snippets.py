import re


def insert_code_snippet(markdown_path, output_path):
    # Read the Markdown file
    with open(markdown_path, "r") as md_file:
        markdown_content = md_file.read()

    # Define the placeholder pattern
    pattern = r"<!-- INSERT CODE: (.+?) -->"

    # Function to replace the placeholder with the code snippet
    def replace_with_code(match):
        code_file_path = match.group(1)
        try:
            with open(code_file_path, "r") as code_file:
                code_content = code_file.read()
            # Return the code block wrapped in Markdown code block syntax
            return f"```python\n{code_content}\n```"
        except FileNotFoundError:
            return f"<!-- ERROR: File not found: {code_file_path} -->"

    # Replace all placeholders with the respective code snippets
    updated_content = re.sub(pattern, replace_with_code, markdown_content)

    # Write the updated content to the output file
    with open(output_path, "w") as output_file:
        output_file.write(updated_content)
