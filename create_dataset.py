import argparse
import glob
import io
import json
import os
import subprocess
import tokenize
from collections import defaultdict
from io import BytesIO

import tqdm
from torch.utils._get_clean_triton import get_clean_triton


def remove_extraneous_newlines(file_path):
    """
    Read the lines from the given Python file, and remove any "extra" blank lines
    so that consecutive blank lines collapse into just one.

    1. Read all lines from the file.
    2. Track whether the previous line was blank.
    3. Only append a blank line to the new list of lines if the previous line wasn't blank.
    4. Return the adjusted code as a string.
    """
    with open(file_path, "r") as f:
        lines = f.readlines()

    cleaned_lines = []
    prev_line_blank = False

    for line in lines:
        if line.strip() == "":
            # This line is blank
            if not prev_line_blank:
                # Keep one blank line if the previous line wasn't blank
                cleaned_lines.append(line)
            # Mark that we've encountered a blank line
            prev_line_blank = True
        else:
            # This line isn't blank, so just keep it
            cleaned_lines.append(line)
            # Reset the blank line marker
            prev_line_blank = False

    return "".join(cleaned_lines)


def remove_python_comments(source: str) -> str:
    """
    Remove all comments from a Python source code string without altering other formatting.

    This function uses the built-in tokenize module to break the source code
    into tokens, then reconstructs the source code while omitting any token
    of type COMMENT. It carefully adds back any whitespace or newlines that occur
    between tokens, so that the formatting of the remaining code is preserved.
    """

    # Encode the source to bytes and create a BytesIO stream for tokenization.
    source_bytes = source.encode("utf-8")
    stream = BytesIO(source_bytes)

    # Initialize the token generator.
    tokens = tokenize.tokenize(stream.readline)

    # We'll rebuild the source using pieces accumulated in this list.
    result = []
    # Keep track of the position (line, column) of the end of the last token added.
    last_lineno, last_col = 1, 0

    for token in tokens:
        token_type = token.type
        token_string = token.string
        start_line, start_col = token.start
        end_line, end_col = token.end

        # Skip the encoding and endmarker tokens.
        if token_type in (tokenize.ENCODING, tokenize.ENDMARKER):
            continue

        if token_type == tokenize.COMMENT:
            # Instead of outputting the comment, update the current position.
            # This has the effect of “removing” the comment along with any space that was
            # solely part of the comment region.
            last_lineno, last_col = end_line, end_col
            continue

        # If there is a gap between the last token and the current token,
        # fill it in (this preserves spaces and newlines from the original source).
        if start_line > last_lineno:
            # Add newlines for any skipped lines.
            result.append("\n" * (start_line - last_lineno))
            # After a newline, reset column to 0.
            last_col = 0

        # Add any extra spaces needed to reach the token’s start column.
        if start_col > last_col:
            result.append(" " * (start_col - last_col))

        # Append the current token’s text.
        result.append(token_string)
        # Update the last position to the end of the current token.
        last_lineno, last_col = end_line, end_col

    # Join all pieces and return the reconstructed source.
    return "".join(result)


def extract_output_code(dir_path):
    # read the file line by line and look for lines that look like
    # [stuff] Output code written to: /tmp/torchinductor_sahanp/y4/cy4ujkrhmfu3t5xkvy53nswxoy5b7he2246t2rrxutae4ksl3dfe.py
    uuid_to_code = defaultdict(set)
    dataset = []
    clean_code_links = defaultdict(
        lambda: defaultdict(lambda: [])
    )  # repo_name, module_name, code_file
    code_files = []
    for file_path in glob.glob(os.path.join(dir_path, "*.txt")):
        with open(file_path, "r") as f:
            for line in f:
                if "Output code written to:" in line:
                    name_string = file_path
                    # remove test_ and .txt
                    # strip dir_path
                    name_string = name_string.split("/")[-1]
                    name_string = name_string[5:]  # remove test_
                    name_string = name_string[:-4]  # remove .txt
                    name_string = name_string.split(".")
                    repo_name = name_string[0]
                    module_name = name_string[1]

                    code_file = line.split("Output code written to:")[1].strip()
                    clean_code_links[repo_name][module_name].append(code_file)
                    code_files.append(code_file)
    with open("clean_code_links.json", "w") as f:
        json.dump(clean_code_links, f, indent=4)
    count = 0
    for repo_name in tqdm.tqdm(clean_code_links.keys(), desc="Cleaning code"):
        if count >= 10:
            break
        for module_name in clean_code_links[repo_name].keys():
            for code_file in clean_code_links[repo_name][module_name]:
                file_name = f"{repo_name}.{module_name}.py"
                if not os.path.exists(f"cleaned_triton/{file_name}"):
                    try:
                        os.environ["TORCHINDUCTOR_DUMP_LAUNCH_PARAMS"] = "1"
                        subprocess.run(["python", code_file])
                        get_clean_triton(code_file, f"cleaned_triton/{file_name}")
                        count += 1
                    except Exception as e:
                        print(f"Failed to clean triton code for {file_name}: {e}")
                        continue

    # for uuid, code_file in tqdm.tqdm(
    #     uuid_to_clean_code.items(), desc="cleaning dataset"
    # ):
    bad_files = []
    # for uuid, code_file in tqdm.tqdm(uuid_to_clean_code.items(), desc="linting code"):
    # try:
    # commentless_code = remove_python_comments(code_file)
    # with open(code_file, "w") as f:
    #     f.write(commentless_code)
    # clean_code = remove_extraneous_newlines(code_file)
    # with open(code_file, "w") as f:
    #     f.write(clean_code)
    # remove unused imports and variables

    # subprocess.run(
    #     [
    #         "autoflake",
    #         "--in-place",
    #         "--remove-all-unused-imports",
    #         "--remove-unused-variables",
    #         code_file,
    #     ]
    # )
    # except Exception as e:
    #     # print(f"Failed to clean triton code for {uuid}: {e}")
    #     bad_files.append(uuid)

    for uuid, code_file in tqdm.tqdm(
        uuid_to_clean_code.items(), desc="Creating dataset"
    ):
        if uuid in bad_files:
            continue
        else:
            generated_code_file = f"generated/random_torch_{uuid}.py"
            with open(generated_code_file, "r") as f:
                generated_code = f.read()
            with open(code_file, "r") as f:
                triton_code = f.read()
            with open(code_file, "w") as f:
                f.write(triton_code)
            dataset.append(
                {
                    "uuid": uuid,
                    "pytorch_code": generated_code,
                    "triton_code": triton_code,
                }
            )
    print(f"Found {len(dataset)} entries")
    return dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_dir_path", type=str, default="generated")
    parser.add_argument("--output_file", type=str, default="dataset.json")
    parser.add_argument("--uuid_file", type=str, default="filtered_uuids.json")
    args = parser.parse_args()

    intermediate_output_path = "inductor_logs"
    # compile_from_folder(args.gen_dir_path, args.uuid_file, intermediate_output_path)

    dataset = extract_output_code(intermediate_output_path)
    with open(args.output_file, "w") as f:
        json.dump(dataset, f, indent=4)


if __name__ == "__main__":
    main()
