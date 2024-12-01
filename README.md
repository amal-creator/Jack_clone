# Jack_clone

# Tweet Generator & Poster using AI

This Python script generates tweets based on a provided blog post using AI and automatically posts them on Twitter. 
It uses the **Transformers** library for generating content and 
the **Tweepy** library for interacting with the Twitter API.

## Features

- **AI-Powered Tweet Generation:** Converts blog content into creative tweets.
- **Batch Tweet Posting:** Automatically posts up to 10 tweets at once.
- **Automatic Reply Handling:** Monitors and responds to replies to Jack's tweets.
- **Rate Limiting and Error Handling:** Handles API rate limits and retries after the specified delay.

## Prerequisites

Before running the script, you will need the following:

- Python 3.8 or later
- [Tweepy](https://www.tweepy.org/) (Twitter API wrapper)
- [Hugging Face Transformers](https://huggingface.co/transformers/) (for model-based text generation)
- [Torch](https://pytorch.org/) (for deep learning)

## Installation

### Step 1: Clone the repository

```bash
git clonehttps://github.com/amal-creator/Jack_clone.git
cd tweet-generator


```

### Step 2: Open in VS Code

Open the project folder in Visual Studio Code:

```bash
code .
```
### Step 3: Install required packages

You can install the required dependencies using `pip`. It is recommended to set up a **virtual environment** first.

```bash
# Create a virtual environment (optional)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

If you don't have a `requirements.txt`, 

### Step 4: Configure your Twitter API keys

You need to create a `.env` file to store your Twitter API keys. You can get these keys by creating a developer account on [Twitter Developer](https://developer.twitter.com/en/apps) and generating your app's API credentials.

Create a `.env` file in the root directory and add your keys:

```
TWITTER_API_KEY=your_twitter_api_key
TWITTER_API_SECRET_KEY=your_twitter_api_secret_key
TWITTER_ACCESS_TOKEN=your_twitter_access_token
TWITTER_ACCESS_TOKEN_SECRET=your_twitter_access_token_secret
```

### Step 5: Create or Edit the `.env` File

Make sure the `.env` file looks like the following:

```
TWITTER_API_KEY={"your-api-key-here"}
TWITTER_API_SECRET_KEY={"your-api-secret-here"}
TWITTER_ACCESS_TOKEN={"your-access-token-here"}
TWITTER_ACCESS_TOKEN_SECRET={"your-access-token-secret-here"}
```

### Step 6: View Output

The script will:
- Read a blog sentence (or set of sentences).
- Generate a series of tweets.
- Post them to Twitter via your connected account.
- Respond to replies to your posts in Jack's voice.

### Optional: Customize the Script

You can adjust various settings such as:
- **Number of tweets per blog post:** Adjust the `num_tweets` parameter to generate a different number of tweets per post.
- **Max tweet length:** Tweak the `max_length` parameter in the `generate_tweet` function.
- **Retry timing:** Adjust the retry time in case the rate limit is exceeded by modifying `time.sleep(15 * 60)`.

## Troubleshooting

### Common Issues

1. **API Rate Limiting:**
   If you hit the rate limit, the script will automatically retry after a 15-minute delay. You can modify this delay by adjusting the `time.sleep(15 * 60)` in the `post_tweets` function.


2. **Invalid API Keys:**
   Ensure that your `.env` file has correct values. Double-check your keys from the [Twitter Developer Portal](https://developer.twitter.com/).

3. **Missing Dependencies:**
   If you encounter import errors, ensure that all required libraries are installed using `pip install -r requirements.txt`.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---



