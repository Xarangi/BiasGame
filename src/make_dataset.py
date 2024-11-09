import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from scenario import get_scenario
from personas import get_attributes, create_backstory
from environment import create_env

# List to store the final dataset
final_dataset = []
dataset_lock = threading.Lock()  # To safely append to final_dataset in a multi-threaded environment

def load_json_lines(file_path):
    """
    Loads a JSON Lines file (newline-delimited JSON) and returns a list of dictionaries.
    Each line in the file is expected to be a valid JSON object.
    """
    data = []
    try:
        with open(file_path, 'r') as json_file:
            for line in json_file:
                if line.strip():  # Only process non-empty lines
                    data.append(json.loads(line))
        return data
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return None


def filter_neg_questions(data):
    """
    Filters the JSON data to retain only one 'neg' question per question_index.
    For each relevant entry, calls the `make_dataset` function in parallel.
    """
    filtered_data = {}

    with ThreadPoolExecutor() as executor:
        futures = []
        
        # Iterate through the data
        for entry in data:
            question_index = entry.get("question_index")
            question_polarity = entry.get("question_polarity")
            
            # Check if the polarity is 'neg' and we haven't yet stored this question_index
            if question_polarity == "neg" and question_index not in filtered_data:
                # Extract the relevant fields
                stereotyped_groups = entry.get("additional_metadata", {}).get("stereotyped_groups", [])
                context = entry.get("context", "")
                answer_info = entry.get("answer_info", {})
                names_list = []
                
                # Iterate through answer_info to gather all names in ans0, ans1, ans2
                for ans_key in ["ans0", "ans1", "ans2"]:
                    if ans_key in answer_info:
                        # Extend names_list with all names in the list for ans0, ans1, ans2
                        names_list.extend(answer_info[ans_key])
        
                # Remove duplicates in the names_list
                names_list = list(set(names_list))

                # Remove each name from the context
                for name in names_list:
                    # Replace the name in the context with an empty string
                    context = context.replace(name, "")
                question = entry.get("question", "")
                
                # Submit the task to the executor for parallel execution
                futures.append(executor.submit(make_dataset, stereotyped_groups, context, question))
                
                # Track unique question_index to avoid duplicates
                filtered_data[question_index] = True
        
        # Wait for all the tasks to complete
        for future in as_completed(futures):
            try:
                future.result()  # This raises any exception that occurred inside the thread
            except Exception as e:
                print(f"An error occurred during dataset creation: {e}")


def make_dataset(stereotyped_groups, context, question):
    """
    Simulates the process of creating a dataset by calling various functions in sequence.
    This function takes the stereotyped_groups, context, and question and processes them
    using the imported functions from scenario, personas, environment, and conversation.
    """
    print(context)
    for group in stereotyped_groups:
        scenario_description, qn = get_scenario(context, question, group)
        print(f"Scenario Description: {scenario_description}")

        env = create_env(scenario_description)
        print(f"Environment: {env}")

        personas = create_backstory(group, stereotyped_groups, env, scenario_description)
        print(f"Personas: {personas}")

        # Safely append the resulting data to the final dataset list using a lock
        with dataset_lock:
            final_dataset.append({
                "group": group,
                "scenario": scenario_description,
                "question": qn,
                "personas": personas
            })


def save_final_dataset(output_file_path):
    """
    Saves the final dataset into a JSON file.
    """
    try:
        with open(output_file_path, 'w') as outfile:
            json.dump(final_dataset, outfile, indent=4)
        print(f"Final dataset has been saved to '{output_file_path}'")
    except IOError as e:
        print(f"Error writing to file {output_file_path}: {e}")


def main(input_file, output_file):
    """
    Main function to load JSON lines, process them, and save the final dataset.
    """
    # Load JSON data from file
    data = load_json_lines(input_file)
    
    if data is None:
        return
    
    # Filter the data and call make_dataset for relevant entries
    filter_neg_questions(data)
    
    # Save the final dataset to a file
    save_final_dataset(output_file)


if __name__ == "__main__":
    # Example usage
    input_file = 'Race_ethnicity.jsonl'  # Replace with the actual input file name (in JSON Lines format)
    output_file = 'final_datasetx.json'  # Replace with the desired output file name
    
    # Call the main function
    main(input_file, output_file)
