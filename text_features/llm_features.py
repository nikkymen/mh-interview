import os
import pandas as pd
import json
import time
import argparse

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
        )

        response_content = completion.choices[0].message.content

        print(response_content)

        if response_content:
            result = {}
            for line in response_content.strip().split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip().lower()
                    if value == "poor":
                        int_value = 0
                    elif value == "not poor":
                        int_value = 1
                    elif value == "high":
                        int_value = 1
                    elif value == "not high":
                        int_value = 0
                    else:
                        int_value = -1

                    result[key] = int_value
            return result

        # if response_content:
        #     return json.loads(response_content)

        return None

    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    finally:
        print("-" * (len(model_name) + 22))
        print("\n")

def extract_tf_llm(prompt: str, model_name: str) -> pd.DataFrame:
    # TODO
    return pd.DataFrame()

def main():

    parser = argparse.ArgumentParser(description='Detection train')

    parser.add_argument('--model', type=str, default='qwen/qwen3-4b:free', help='model id')
    parser.add_argument('--key', type=str, default='', help='openrouter API key')
    parser.add_argument('--sleep', type=float, default=0, help='sleep between runs')

    args = parser.parse_args()

    if args.key:
        api_key = args.key
    else:
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

    # Used:
    # qwen-2.5-72b-instruct:free
    # meta-llama/llama-3.3-70b-instruct:free
    # openai/gpt-oss-120b

    # Open:

    # qwen/qwen3-4b:free
    # qwen/qwq-32b:free
    # mistralai/mistral-small-3.2-24b-instruct:free
    # openai/gpt-oss-20b:free
    # deepseek/deepseek-r1-0528:free
    # meta-llama/llama-3.3-70b-instruct:free

    # Proprietary:

    # openai/gpt-5
    # google/gemini-2.5-pro
    # x-ai/grok-4
    # mistralai/mistral-large
    # deepseek/deepseek-r1

    # Not supported:
    # openai/o3-pro # OpenAI is requiring a key to access this model
    # anthropic/claude-opus-4.1

    # Read the prompt from the file
    with open('system_prompt.txt', 'r') as file:
        system_prompt = file.read().strip()

    output_path = "predictions2/" + args.model.split('/')[1].split(':')[0] + ".csv"

    Path(output_path).parent.mkdir(exist_ok=True)

    # Get directory path with prompt files
    prompts_dir = "transcripts"

    if not os.path.exists(prompts_dir):
        print(f"Error: Directory '{prompts_dir}' does not exist.")
        return

    # Get all text files from the directory
    text_files = [f for f in os.listdir(prompts_dir) if f.endswith('.txt')]
    text_files.sort()

    #text_files = ['9234.txt']

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

        # Get model predictions
        prediction = query(client, args.model, system_prompt, prompt)

        if prediction:
            row_data = {'id': id}

            # Add model-specific prefix to each parameter
            for param, value in prediction.items():
                row_data[param] = value

            results.append(row_data)

        # Save progress after each file is processed
        df = pd.DataFrame(results)

        # Save to CSV (append mode if file exists)
        df.to_csv(output_path, index=False, mode='w')

        if args.sleep:
            time.sleep(args.sleep)  # To avoid hitting rate limits

if __name__ == "__main__":
    main()
