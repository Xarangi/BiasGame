import openai

# Replace with your actual OpenAI API key
def create_env(scenario):
    openai.api_key = os.get_env("GPT_KEY")
    prompt = (
        "Create some common memories of general facts and the environment that will be common for all involved in the following scenario. Examples include name of location, time (if relevant) and any other key factors:"
        f"{scenario}"
    )

    response = openai.chat.completions.create(
        model='gpt-4o-2024-08-06',
        messages=[
            {"role": "system", "content": "You are an environment creation assistant that creates general and common memories of occurrences that make a given scenario more realistic. You don't provide any other details aside from the common memories presented as simple text (no more than 5 memories)"},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
        max_tokens=1200,
    )
    env = response.choices[0].message.content.strip()
    return env
