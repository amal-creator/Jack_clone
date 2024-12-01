import os
import requests
import torch
import random
import pandas as pd
import time
from bs4 import BeautifulSoup
from textblob import TextBlob
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
import tweepy
from dotenv import load_dotenv
import emoji
load_dotenv()

# Twitter API credentials

API_KEY = os.getenv("TWITTER_API_KEY")
API_SECRET_KEY = os.getenv("TWITTER_API_SECRET_KEY")
ACCESS_TOKEN = os.getenv("TWITTER_ACCESS_TOKEN")
ACCESS_TOKEN_SECRET = os.getenv("TWITTER_ACCESS_TOKEN_SECRET")
BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN")

# Authenticate with Twitter

client = tweepy.Client(
    bearer_token=BEARER_TOKEN,
    consumer_key=API_KEY,
    consumer_secret=API_SECRET_KEY,
    access_token=ACCESS_TOKEN,
    access_token_secret=ACCESS_TOKEN_SECRET,
)

# Step 1: Scrape Blog Content


def scrape_blog(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    posts = soup.find_all([["h1", "h2", "p"]])  
    texts = [post.get_text(strip=True) for post in posts]
    
    return texts

jack_blog_url = "https://jackjay.io/"
jack_blog_content = scrape_blog(jack_blog_url)


df_blog = pd.DataFrame(jack_blog_content, columns=['text'])
df_blog.to_csv('jack_blog_content.csv', index=False)
print("Blog content saved to jack_blog_content.csv")


# Save scraped content


df_blog = pd.DataFrame(jack_blog_content, columns=['text'])
df_blog.to_csv('jack_blog_content.csv', index=False)
print("Blog content saved to jack_blog_content.csv")

# Step 2: Add Data from `jackio.csv` and Merge


jackio_linkdin_data = pd.read_csv('jack.io.csv', encoding='latin-1')  
df_combined = pd.concat([df_blog, jackio_linkdin_data], ignore_index=True)  


# Save merged content 


df_combined.to_csv('jack_combined_content.csv', index=False)
print("Combined content saved to jack_combined_content.csv")

# Step 3: Clean the Combined Data


def clean_text(text):
    text = text.strip()  
    text = ' '.join(text.split())  
    text = ''.join(e for e in text if e.isalnum() or e.isspace() or e in ['.', ',', '?', '!'] or  emoji.is_emoji(e))  
    return text

df_combined['text'] = df_combined['text'].astype(str).apply(clean_text)  
df_cleaned = df_combined.drop_duplicates(subset=['text']).dropna(subset=['text'])  

# Save cleaned content

df_cleaned.to_csv('jack_cleaned_content.csv', index=False)
print("Cleaned content saved to jack_cleaned_content.csv")


# Step 2: Sentiment Analysis

def get_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

df_cleaned['sentiment'] = df_cleaned['text'].apply(get_sentiment)  
df_cleaned.to_csv('jack_cleaned_with_sentiment.csv', index=False)
print("Sentiment analysis added and saved.")


# Step 3: Fine-tune GPT-2

tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")
tokenizer.pad_token = tokenizer.eos_token

train_texts = df_cleaned['text'].tolist()
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512,return_tensors="pt")



input_ids = train_encodings['input_ids']
attention_mask = train_encodings['attention_mask']

class BlogDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = item['input_ids'].clone()
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])

# Split the dataset into training and evaluation sets


train_size = int(0.8 * len(train_texts))
eval_texts = train_texts[train_size:]
train_texts = train_texts[:train_size]

# Tokenize the evaluation texts


eval_encodings = tokenizer(eval_texts, truncation=True, padding=True, max_length=512,return_tensors="pt")
eval_dataset = BlogDataset(eval_encodings)

# Update training arguments


training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=4,
    per_device_train_batch_size=4,
    learning_rate=3e-5,
    save_steps=1000,
    save_total_limit=2,
    eval_strategy="steps",  
    eval_steps=200,  
    logging_dir='./logs',
     logging_steps=50,
)


# Tokenize the training texts
TRAIN_MODEL = True

if TRAIN_MODEL:
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
    train_dataset = BlogDataset(train_encodings)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,  
        eval_dataset=eval_dataset,   
    )



    trainer.train()
    model.save_pretrained('./Jack_model_gpt2')
    print("Model fine-tuned and saved.")

else:
    if not os.path.exists('./Jack_model_gpt2'):
        raise FileNotFoundError("Model not found")

# Step 4: Generate and Post Tweets

model = GPT2LMHeadModel.from_pretrained('./Jack_model_gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token



def preprocess_blog_content(file_path):
    df = pd.read_csv(file_path,encoding='latin-1')
    sentences = []
    for text in df['text']:
        for sentence in text.split('.'):
            sentence = sentence.strip()
            if 10 < len(sentence) <= 120:
                sentences.append(sentence)
    return sentences






blog_sentences = preprocess_blog_content('jack_blog_content.csv')
def generate_tweet(prompt, num_tweets=1):
    #print(f"Generating tweet for prompt:{prompt}")
    generated_tweets = []
    for _ in range(num_tweets):
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        attention_mask = torch.ones(input_ids.shape, device=input_ids.device) 
        seed = random.randint(0, 10000)
        torch.manual_seed(seed)
        # Ensure input is within model's context window
        #max_positions =  getattr(model.config,"n_positions",None)
        #if max_positions is None:
         #   raise ValueError("Model config desnt have n_positn")
        #i#f len(input_ids[0]) > model.config.n_positions:
         #   input_ids = input_ids[:, :model.config.n_positions]

        #print(f"input ids (tokenized): {input_ids}")
        
        
        output = model.generate(
            input_ids,
            attention_mask=attention_mask,  
            do_sample=True,
            max_length=120,
            num_return_sequences=num_tweets,
            pad_token_id=tokenizer.eos_token_id,
            temperature=1.,
            top_k=50,
            top_p=0.95,
            no_repeat_ngram_size=2,
        )

        #print(f"Generated outpt:{output}")
        
        tweet = tokenizer.decode(output[0],skip_special_tokens=True).strip() 
        if 10 < len(tweet) <= 280:
            generated_tweets.append(tweet)

    #print(f"generated tweet: {generated_tweets}")
    return generated_tweets



def post_tweets(blog_sentences,max_posts = 10):
    posted_tweets = set()
    retry_attempts = 0
    posted_count = 0
    max_posts = 10

    for prompt in blog_sentences:
        if posted_count >= max_posts:
            break
        try:
            generated_tweets = generate_tweet(prompt, num_tweets=1)
            #print(f"Generated tweet: {generated_tweets}")
            
            for tweet in generated_tweets:
                if posted_count >= max_posts:
                    break
                if tweet in posted_tweets:
                    continue
                
                response = client.create_tweet(text=tweet)
                
                print(f"Posted tweet: {tweet}")
                posted_tweets.add(tweet)
                posted_count += 1
                
                time.sleep(2)
        except tweepy.errors.TooManyRequests as e:
            retry_attempts += 1
            print(f"Rate limit exceeded. Waiting for {e.response.headers['x-rate-limit-reset']} seconds (Attempt {retry_attempts})...")
            reset_time = int(e.response.headers['x-rate-limit-reset']) 
            wait_time = reset_time - time.time()  
            if wait_time > 0:
                time.sleep(wait_time + 5) 
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    print(f"Total tweets posted: {posted_count}")




blog__sentences = ["I’m proud to share that I am working as a core member of an innovative AI content generation project at Persist Ventures, under the inspiring leadership of Jackson Jesionowski. This journey has been an incredible opportunity to explore cutting-edge technologies and creative solutions. 🚀",]  


post_tweets(blog_sentences,max_posts=10)