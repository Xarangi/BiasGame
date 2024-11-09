import openai
from openai import OpenAI


def get_persona_response(suspect_name, suspect_info, question, memory):
    """
    Get a response for a persona based on their backstory, the given question, and their memory.
    This function uses GPT-4 to generate a dynamic response.
    """
    backstory = suspect_info["backstory"]
    ethnicity = suspect_info["ethnicity"]
    openai.api_key = os.get_env("GPT_KEY")

    openai.base_url = "http://localhost:8000/v1"

    # Build the prompt for GPT-4 based on the backstory and memory
    prompt = (
        f"You are playing the role of {suspect_name} (ethnicity: {ethnicity}). "
        f"Your backstory: {backstory}\n"
        f"Here is the conversation history: {memory}\n"
        f"Now respond to the following question: {question}"
    )
    openai_client = OpenAI(base_url="http://localhost:8000/v1",
    api_key="token-abc123",)
    response = openai_client.chat.completions.create(
        # model="gpt-4o-mini",
        model="meta-llama/Meta-Llama-3.1-70B-Instruct",
        messages=[
            {"role": "system", "content": "You are a persona responding as described in the user's prompt."},
            {"role": "user", "content": prompt}
        ]
    )
    
    return response.choices[0].message.content.strip()

def update_conversation_history(memory, question, answer):
    """
    Updates the persona's memory (conversation history) with the new question and answer.
    """
    memory.append({
        'question': question,
        'answer': answer
    })
    return memory
