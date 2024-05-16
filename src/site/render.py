import os

import snippets


def main():
    md_files = [f for f in os.listdir("./src/site") if f.endswith(".md")]
    for file in md_files:
        snippets.insert_code_snippet(f"./src/site/{file}", f"./{file}")


if __name__ == "__main__":
    main()
