# convert_json_to_txt.py
import json
import argparse
import os

def convert(json_file_path, txt_file_path):
    """
    Converts a JSON file containing prediction outputs to a TXT file
    with one output string per line.
    """
    if not os.path.exists(json_file_path):
        print(f"Error: JSON file not found at {json_file_path}")
        return

    try:
        with open(json_file_path, 'r', encoding='utf-8') as f_json:
            data = json.load(f_json)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {json_file_path}")
        return
    except Exception as e:
        print(f"An error occurred while reading {json_file_path}: {e}")
        return

    if not isinstance(data, list):
        print(f"Error: Expected a list of items in JSON file {json_file_path}, but got {type(data)}")
        return

    print(f"Converting {json_file_path} to {txt_file_path}...")
    with open(txt_file_path, 'w', encoding='utf-8') as f_txt:
        for item in data:
            if isinstance(item, dict) and 'output' in item:
                f_txt.write(item['output'] + "\n")
            else:
                print(f"Warning: Skipping item due to missing 'output' key or incorrect format: {item}")
    
    print(f"Conversion successful. TXT file saved to {txt_file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert prediction JSON to TXT format.")
    parser.add_argument("json_file", help="Path to the input JSON prediction file.")
    parser.add_argument("txt_file", help="Path to the output TXT file.")
    
    args = parser.parse_args()
    
    convert(args.json_file, args.txt_file)