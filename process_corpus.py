import json
import os
from convokit import Corpus, download


def create_formatted_dataset(corpus, output_file):
    """
    Creates a dataset with formatted conversation data suitable for fine-tuning or inference.
    :param corpus: Convokit Corpus object.
    :param output_file: Path to save the formatted dataset.
    """
    formatted_conversations = []

    for conversation in corpus.iter_conversations():
        # Extract conversation-level metadata
        metadata = conversation.meta
        title = metadata.get("title", "Unknown Title")
        subreddit = metadata.get("subreddit", "Unknown Subreddit")

        # Format conversation metadata
        conversation_data = {
            "Conversation Metadata": {
                "Title": title,
                "Subreddit": subreddit
            },
            "Conversation": []
        }

        # Build utterance map for reply structure
        utterance_map = {utterance.id: utterance for utterance in conversation.iter_utterances()}

        # Iterate over utterances and format
        for utterance in conversation.iter_utterances():
            speaker = utterance.speaker.id
            text = utterance.text
            reply_to_id = utterance.reply_to
            reply_to_speaker = (
                utterance_map[reply_to_id].speaker.id if reply_to_id and reply_to_id in utterance_map else "None"
            )

            # Add formatted utterance
            conversation_data["Conversation"].append({
                "Speaker": speaker,
                "Replies to": reply_to_speaker,
                "Message": text
            })

        formatted_conversations.append(conversation_data)

    # Save the formatted dataset
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(formatted_conversations, f, indent=4, ensure_ascii=False)
    print(f"Formatted dataset saved to {output_file}")


# Define and download subreddit corpus
output_dir = "../../data/tokenized_training_inputs/v2"
os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists

subreddits = ["GenZ", "AskHistorians", "askscience", "business", "economy", "personalfinance", "tech"]

for subreddit in subreddits:
    output_file = os.path.join(output_dir, f"{subreddit}-tokenized_inputs.json")

    # Check if the file already exists
    if os.path.exists(output_file):
        print(f"Skipping {subreddit} - already processed.")
        continue

    print(f"Processing subreddit: {subreddit}")
    try:
        corpus = Corpus(download(f'subreddit-{subreddit}'))
        create_formatted_dataset(corpus, output_file)
    except Exception as e:
        print(f"Error processing {subreddit}: {e}")
