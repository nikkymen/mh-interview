import os
import pandas as pd
import json

from pathlib import Path
from openai import OpenAI


def query(client: OpenAI, model_name: str, system_prompt: str, prompt: str):
    """
    Sends a prompt to a specified model via the OpenRouter API and prints the response.
    """
    print(f"--- Querying model: {model_name} ---")
    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            extra_headers={
                "X-Title": "MH-Interview-App",
            },
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "parameters",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "WHO-5": {
                                "type": "integer",
                                "description": "Well-being Index",
                                "minimum": 1,
                                "maximum": 3
                            },
                            "PSS-4": {
                                "type": "integer",
                                "description": "Perceived Stress Scale",
                                "minimum": 1,
                                "maximum": 3
                            },
                            "GAD-7": {
                                "type": "integer",
                                "description": "Generalized Anxiety Disorder",
                                "minimum": 1,
                                "maximum": 4
                            },
                            "PHQ-9": {
                                "type": "integer",
                                "description": "Patient Health Questionnaire - Depression",
                                "minimum": 1,
                                "maximum": 5
                            },
                            "Alienation": {
                                "type": "integer",
                                "description": "Degree of Alienation from Studies",
                                "minimum": 1,
                                "maximum": 3
                            },
                            "Burnout": {
                                "type": "integer",
                                "description": "Degree of Burnout",
                                "minimum": 1,
                                "maximum": 3
                            },
                        },
                        "required": ["WHO-5", "PSS-4", "GAD-7", "PHQ-9", "Alienation", "Burnout"],
                        "additionalProperties": False,
                    },
                },
            },
        )

        response_content = completion.choices[0].message.content

        print(response_content)

        if response_content:
            return json.loads(response_content)

        return None

    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    finally:
        print("-" * (len(model_name) + 22))
        print("\n")


def main():
    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        print("Error: OPENROUTER_API_KEY environment variable not set.")
        print("Please set it by running: export OPENROUTER_API_KEY='your-key-here'")
        return

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    # https://openrouter.ai/docs#models
    # https://openrouter.ai/settings/integrations

    # llms = [
    #     "qwen/qwen3-4b:free",
    #     "mistralai/mistral-small-3.2-24b-instruct:free"
    #  #   "openai/gpt-oss-20b:free",
    # ]

    # Read the prompt from the file
    with open('system_prompt.txt', 'r') as file:
        system_prompt = file.read().strip()

    # free_llms = [
    #     "deepseek/deepseek-r1-0528:free"
    #     "qwen/qwq-32b:free"
    # ]

    llms = [
         "deepseek/deepseek-r1-0528:free"
    ]

    output_path = "predictions/llms.csv"

    if len(llms) == 1:
        output_path = "predictions/" + llms[0].split('/')[1].split(':')[0] + ".csv"

    Path(output_path).parent.mkdir(exist_ok=True)

    # llms = [
    #     "openai/gpt-5",
    #     "google/gemini-2.5-pro"
    #     "x-ai/grok-4"
    #     "mistralai/mistral-large"
    #     "deepseek/deepseek-r1"
    # ]

    # openai/gpt-5
    # google/gemini-2.5-pro
    # x-ai/grok-4
    # deepseek/deepseek-r1
    # mistralai/mistral-large

    # Not supported:
    # openai/o3-pro # OpenAI is requiring a key to access this model
    # anthropic/claude-opus-4.1

    # Get directory path with prompt files
    prompts_dir = "transcripts"

    if not os.path.exists(prompts_dir):
        print(f"Error: Directory '{prompts_dir}' does not exist.")
        return

    # Get all text files from the directory
    text_files = [f for f in os.listdir(prompts_dir) if f.endswith('.txt')]
    text_files.sort()


    # Check if the output file already exists and load it
    if os.path.exists(output_path):
        existing_df = pd.read_csv(output_path)
        processed_ids = existing_df['id'].tolist()
        processed_ids = [str(vid) for vid in existing_df['id'].tolist()]

        results = existing_df.to_dict('records')
        print(f"Loaded {len(processed_ids)} existing results from {output_path}")
    else:
        processed_ids = []
        results = []

    # Process each text file
    for filename in text_files:
        id = Path(filename).stem

        # Skip files that have already been processed
        if id in processed_ids:
            print(f"Skipping already processed: {filename}")
            continue

        filepath = os.path.join(prompts_dir, filename)

        # Read the prompt from the file
        with open(filepath, 'r') as file:
            prompt = file.read().strip()

        print(f"\nProcessing: {filename}")

        # Initialize row with filename
        row_data = None

        # Query each model and add results to the row
        for model in llms:
            #model_key = model.split('/')[1].split(':')[0]  # Extract clean model name

            # Get model predictions
            prediction = query(client, model, system_prompt, prompt)

            if prediction:
                row_data = {'id': id}

                # Add model-specific prefix to each parameter
                for param, value in prediction.items():
                    #row_data[f"{param}__{model_key}"] = value
                    row_data[param] = value

        # Add this row to results
        if row_data:
            results.append(row_data)

        # Save progress after each file is processed
        df = pd.DataFrame(results)

        # Save to CSV (append mode if file exists)
        df.to_csv(output_path, index=False, mode='w')

if __name__ == "__main__":
    main()
