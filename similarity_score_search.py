import praw
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import string
import json
import time

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Configure paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(BASE_DIR, "model_training/trained_models/v2")
OUTPUT_FILE = os.path.join(BASE_DIR, "app_data/v3/recent_summaries.json")

# Initialize Reddit API
reddit = praw.Reddit(
    client_id="rYPeaC9kxCTjPjE_DCw-QQ",
    client_secret="az_2SpHuFT5mC97U5UraWubWoBqwtA",
    user_agent="themightytak"
)

# Load fine-tuned model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)


# Preprocessing query with stemming and lemmatization
def preprocess_query(query):
    stop_words = set(stopwords.words("english"))
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    tokens = nltk.word_tokenize(query.lower())
    processed_tokens = [
        lemmatizer.lemmatize(stemmer.stem(token))
        for token in tokens
        if token not in stop_words and token not in string.punctuation
    ]
    return " ".join(processed_tokens)


# Search Reddit posts
def search_posts(query, subreddit_name=None, num_posts=10):
    processed_query = preprocess_query(query)  # Apply query preprocessing
    print(f"Searching for '{processed_query}'...")

    if subreddit_name:
        subreddit = reddit.subreddit(subreddit_name)
        search_results = subreddit.search(processed_query, limit=num_posts)
    else:
        search_results = reddit.subreddit("all").search(processed_query, limit=num_posts)

    posts = []
    for post in search_results:
        posts.append({
            "title": post.title,
            "selftext": post.selftext,
            "url": post.url
        })
    return posts


# Summarize posts
def summarize_posts(post):
    if not post.get("selftext") or post["selftext"].startswith("http"):
        return {
            "title": post["title"],
            "summary": "No meaningful content to summarize.",
            "url": post["url"]
        }

    input_text = f"Title: {post['title']}\n\nContent: {post['selftext']}"
    inputs = tokenizer(input_text, max_length=512, truncation=True, return_tensors="pt")
    summary_ids = model.generate(inputs["input_ids"], max_length=150, min_length=30, length_penalty=2.0, num_beams=4,
                                 early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return {
        "title": post["title"],
        "summary": summary,
        "url": post["url"]
    }


# User interface for searching and summarizing
def user_interface():
    start_time = time.time()  # Start timing
    print("Welcome to the Reddit Summarizer!")
    query = input("Enter your search query: ").strip()
    subreddit_name = input("Enter the subreddit name (leave blank for all subreddits): ").strip()
    num_posts = int(input("Enter the number of posts to summarize: ").strip())

    print(
        f"\nSearching for '{query}' in {'all subreddits' if not subreddit_name else f'subreddit: {subreddit_name}'}...\n")
    posts = search_posts(query, subreddit_name=subreddit_name, num_posts=num_posts)

    summaries = []
    for post in posts:
        summary = summarize_posts(post)
        summaries.append(summary)
        print(f"\nTitle: {summary['title']}\nSummary: {summary['summary']}\nURL: {summary['url']}\n")

    # Save results to a file
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(summaries, file, indent=4, ensure_ascii=False)

    elapsed_time = time.time() - start_time  # Calculate elapsed time
    print(f"\nExecution Time: {elapsed_time:.2f} seconds")  # Print elapsed time


if __name__ == "__main__":
    user_interface()
