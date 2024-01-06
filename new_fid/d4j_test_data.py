import os
import json
import argparse

def get_all_directories(root_dir='.'):
    directories = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    return directories

def read_json_files(directory):
    json_files = [f for f in os.listdir(directory) if f.endswith('.json')]
    data = []

    for json_file in json_files:
        with open(os.path.join(directory, json_file), 'r') as f:
            try:
                json_data = json.load(f)
                # Add directory name to each JSON entity
                for entity in json_data:
                    entity['project'] = directory
                data.extend(json_data)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON in file {json_file}: {e}")

    return data

def combine_and_write_to_file(data, output_file='combined_data.json'):
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

def argument_parser():
    parser = argparse.ArgumentParser(description='Combine JSON files from multiple directories')
    parser.add_argument('--root_dir', type=str, default='.', help='Root directory to search for JSON files')
    parser.add_argument('--output_file', type=str, default='combined_data.json', help='Output file name')
    args = parser.parse_args()
    return args

def main():
    args = argument_parser()
    root_dir = args.root_dir
    output_file = args.output_file

    
    all_directories = get_all_directories(root_dir)

    combined_data = []

    for directory in all_directories:
        directory_path = os.path.join(root_dir, directory)
        directory_data = read_json_files(directory_path)
        combined_data.extend(directory_data)

    combine_and_write_to_file(combined_data,output_file=output_file)

if __name__ == "__main__":
    main()