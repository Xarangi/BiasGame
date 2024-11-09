import openai
import random
import re

def extract_backstory(response_text):
    """
    Extracts the backstory from a response string formatted as either 'Backstory: (backstory)' 
    or 'Backstory: backstory' (with or without parentheses).
    
    Args:
        response_text (str): The response text containing the backstory.
    
    Returns:
        str: The extracted backstory.
    """
    # First try to extract the backstory with parentheses
    backstory_match = re.search(r"Backstory:\s*\((.*?)\)", response_text, re.DOTALL)
    
    if not backstory_match:
        # If no parentheses are found, try without parentheses
        backstory_match = re.search(r"Backstory:\s*(.*)", response_text, re.DOTALL)
    
    if backstory_match:
        # Return the extracted backstory, stripped of leading/trailing spaces
        return backstory_match.group(1).strip()
    
    return None
def extract_name_from_prompt_name(prompt_result):
    """
    Extracts the name from the first prompt result: "Name: (name)".
    """
    name_match = re.search(r"Name:\s*(.+)", prompt_result)
    if name_match:
        return name_match.group(1).strip()
    return None

def extract_names_and_ethnicities(prompt_result, stereotyped_groups):
    """
    Extracts names and ethnicities from the prompt result in the format:
    "Name: (name) Ethnicity: (ethnicity), Name: (name) Ethnicity: (ethnicity)".
    If no parentheses are found, it also handles the format without parentheses.
    
    Args:
        prompt_result (str): The input string containing names and ethnicities.
        stereotyped_groups (list): A list of ethnic groups to be replaced with alternatives.
    
    Returns:
        dict: A dictionary of name and ethnicity pairs.
    """
    # First, attempt to match the format with parentheses
    matches = re.findall(r"Name: \(([^)]+)\)\s*Ethnicity: \(([^)]+)\)", prompt_result)

    if not matches:
        # If no matches are found, try the format without parentheses
        matches = re.findall(r"Name:\s*([^\s,]+(?:\s+[^\s,]+)*)\s*Ethnicity:\s*([^\s,]+(?:\s+[^\s,]+)*)", prompt_result)
    
    name_ethnicity_dict = {}
    
    for match in matches:
        name, ethnicity = match
        
        # Replace the ethnicity if it is in the stereotyped_groups
        if ethnicity.strip() in stereotyped_groups:
            ethnicity_options = ["White", "Caucasian"]
            ethnicity = random.choice(ethnicity_options)
        
        # Add the cleaned name and ethnicity to the dictionary
        name_ethnicity_dict[name.strip()] = ethnicity.strip()
    
    return name_ethnicity_dict

def generate_name_ethnicity_dictionary(prompt_name_result, prompt_result, group, stereotyped_groups):
    """
    Generates the final name and ethnicity dictionary from both prompts.
    """
    # Extract the name from the first prompt result
    name_from_prompt = extract_name_from_prompt_name(prompt_name_result)

    # Extract names and ethnicities from the second prompt result
    name_ethnicity_dict = extract_names_and_ethnicities(prompt_result, stereotyped_groups)

    # Add the name from the first prompt to the dictionary
    if name_from_prompt:
        # You can assign a placeholder ethnicity or process it separately as needed
        name_ethnicity_dict[name_from_prompt] = f'{group}'

    return name_ethnicity_dict
# Replace with your actual OpenAI API key

def get_attributes(group,stereotyped_groups,scenario):
    openai.api_key = os.get_env("API_KEY")

    str_grps = ", ".join(stereotyped_groups)
    prompt_name = (f"Return a name for a person belonging to this ethnic group: {group}"
                    "Format: Name: name")
    prompt = (
        f"Return names and ethnicities for 4 people involved in the given scenario. Strictly ensure that the 4 race/ethnicities do NOT belong to one of the following groups: {str_grps}. Also ensure you return ethnicities/race not nationalities. Finally, ensure that atleast 2 out of the 4 are White/Caucasian."
        f"Scenario: {scenario}"
        "Return your answer in the format: Name: name Ethnicity: ethnicity, Name: name Ethnicity: ethnicity, and so on."

    )

    response = openai.chat.completions.create(
        model='gpt-4o-2024-08-06',
        messages=[
            {"role": "system", "content": "You are a system that returns names and race/ethnicities in a comma separated format, or according to the instructions."},
            {"role": "user", "content": prompt_name},
        ],
        temperature=0.7,
        max_tokens=20,
    )
    name = response.choices[0].message.content.strip()
    response = openai.chat.completions.create(
        model='gpt-4o-2024-08-06',
        messages=[
            {"role": "system", "content": "You are a system that returns names and race/ethnicities in a comma separated format, or according to the instructions."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
        max_tokens=200,
    )
    name_ethn = response.choices[0].message.content.strip()
    print(name_ethn)
    name_ethnicity_dict = generate_name_ethnicity_dictionary(name, name_ethn, group, stereotyped_groups)

    return name_ethnicity_dict

def create_backstory(group, stereotyped_groups, env, context):
    openai.api_key = os.get_env("API_KEY")

    attributes = get_attributes(group,stereotyped_groups,context)
    personas_dict = {}

    for name, ethnicity in attributes.items():
        # Generate a prompt for the backstory creation
        prompt = (
            f"Create a backstory for {name}, who belongs to the {ethnicity} ethnicity. Keep in mind the context of the scenario where {name} is one of the 5 people mentioned, and the common memories relevant for the scenario"
            f"Include details about their upbringing, cultural background, and personal interests. But keep it concise (less than 300 words)"
            f"Scenario: {context}"
            f"Common Memories: {env}"
        )

        # Call to OpenAI API to generate the backstory
        response = openai.chat.completions.create(
            model='gpt-4o-2024-08-06',  # Ensure you're using the correct model
            messages=[
                {"role": "system", "content": "You are a creative writer that creates character backstories. You return your answers in the format: Backstory: (backstory)"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1200,
        )
        
        # Extract the backstory from the API response
        backstory = extract_backstory(response.choices[0].message.content.strip())

        # Store the result in a new dictionary
        personas_dict[name] = {
            "ethnicity": ethnicity,
            "backstory": backstory + f"\nRecent Common Memories: \n{env}"
        }

    return personas_dict