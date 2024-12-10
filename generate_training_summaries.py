import os
import json
import openai

# Set the OpenAI API Key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load JSON file
def load_corpus(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

# Generate a GPT-compatible input prompt
def generate_input_prompt(conversation):
    metadata = conversation["Conversation Metadata"]
    nodes = conversation["Conversation"]

    metadata_text = f"""
### Metadata
- Title: {metadata.get('Title', 'N/A')}
- Subreddit: {metadata.get('Subreddit', 'N/A')}
    """

    node_map = {i: f"{node['Speaker']}: {node['Message']}" for i, node in enumerate(nodes)}
    conversation_text = "\n".join(
        f"{node_map[i]} ↪ {node_map.get(i + 1, '')}" for i in range(len(nodes) - 1)
    )

    return f"""
{metadata_text.strip()}

### Conversation
{conversation_text.strip()}
    """

# Generate summary for a single conversation
def generate_summary(conversation):
    input_prompt = generate_input_prompt(conversation)

    instruction = """
You are a summarization assistant. Your task is to generate a concise and structured summary of a Reddit conversation.
Focus on the key points, main arguments, and overall sentiment expressed in the replies.
    """

    prompt = f"{instruction}\n\n{input_prompt}\n\nProvide a summary of this conversation:"

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a summarization assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
        )
        summary = response["choices"][0]["message"]["content"].strip()
        return summary, prompt
    except Exception as e:
        print(f"Error generating summary: {e}")
        return None, prompt

# Process a single JSON file
def process_file(file_path, detailed_output_path, fine_tuning_output_path, limit):
    corpus = load_corpus(file_path)
    if limit is not None:
        corpus = corpus[:limit]

    detailed_output_data = []
    fine_tuning_data = []

    for i, conversation in enumerate(corpus):
        print(f"Processing conversation {i + 1}/{len(corpus)} in file {os.path.basename(file_path)}...")
        summary, prompt = generate_summary(conversation)

        if summary:
            detailed_output_data.append({
                "conversation_id": i,
                "conversation": conversation,
                "prompt": prompt,
                "summary": summary
            })

            fine_tuning_data.append({
                "input": prompt.strip(),
                "output": summary
            })

    # Save detailed output
    with open(detailed_output_path, "w", encoding="utf-8") as f:
        json.dump(detailed_output_data, f, indent=4, ensure_ascii=False)

    # Save fine-tuning output
    with open(fine_tuning_output_path, "w", encoding="utf-8") as f:
        json.dump(fine_tuning_data, f, indent=4, ensure_ascii=False)

# Process all JSON files in a directory
def process_directory(input_dir, detailed_output_dir, fine_tuning_output_dir, limit):
    if not os.path.exists(detailed_output_dir):
        os.makedirs(detailed_output_dir)
    if not os.path.exists(fine_tuning_output_dir):
        os.makedirs(fine_tuning_output_dir)

    for file_name in os.listdir(input_dir):
        if file_name.endswith(".json"):
            file_path = os.path.join(input_dir, file_name)

            detailed_output_path = os.path.join(detailed_output_dir, f"{os.path.splitext(file_name)[0]}_detailed.json")
            fine_tuning_output_path = os.path.join(fine_tuning_output_dir,
                                                   f"{os.path.splitext(file_name)[0]}_fine_tuning.json")

            # Check if the corresponding output files exist
            if os.path.exists(detailed_output_path) or os.path.exists(fine_tuning_output_path):
                print(f"Skipping file as output already exists: {file_name}")
                continue

            print(f"Processing file: {file_name}")
            process_file(file_path, detailed_output_path, fine_tuning_output_path, limit)

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    default_input_dir = os.path.join(base_dir, "../../data/tokenized_training_inputs/v2")
    default_output_dir = os.path.join(base_dir, "../../data/generated_training_outputs/v2/detailed")
    default_training_data_dir = os.path.join(base_dir, "../../data/generated_training_outputs/v2/fine_tuning")
    default_limit = 50

    print("Starting processing with the following parameters:")
    print(f"Input Directory: {default_input_dir}")
    print(f"Detailed Output Directory: {default_output_dir}")
    print(f"Fine-Tuning Output Directory: {default_training_data_dir}")
    print(f"Limit: {default_limit} conversations per file")

    process_directory(
        input_dir=default_input_dir,
        detailed_output_dir=default_output_dir,
        fine_tuning_output_dir=default_training_data_dir,
        limit=default_limit
    )
