# Tweets Dataset

Place tweet CSV files here. Supported Kaggle datasets:

## Option 1: ChatGPT Tweets (Recommended)
- **Source**: https://www.kaggle.com/datasets/khalidryder777/500k-chatgpt-tweets-jan-mar-2023
- **Download**: Click "Download" on Kaggle, extract the CSV
- **Place here**: `chatgpt_tweets.csv` (or similar)

## Option 2: Twitter Data
- **Source**: https://www.kaggle.com/datasets/smmmmmmmmmmmm/twitter-data
- **Download**: Click "Download" on Kaggle, extract the CSV
- **Place here**: The main CSV file

## Option 3: Russian Troll Tweets (FiveThirtyEight IRA dataset)
- **Source**: https://www.kaggle.com/datasets/vikasg/russian-troll-tweets
- **Download**: 
  ```bash
  curl -L -o ~/Downloads/russian-troll-tweets.zip \
    https://www.kaggle.com/api/v1/datasets/download/vikasg/russian-troll-tweets
  ```
- **Extract** and place CSV(s) here
- **Columns**: content, author, region, language, publish_date, followers, account_category
- **Note**: ~3 million tweets from IRA-linked accounts (2012-2018). Great for exploring troll account categories and regional patterns.

## Optional: dataset.yaml

For explicit column mapping, create `dataset.yaml`:

```yaml
source: kaggle
kaggle_slug: khalidryder777/500k-chatgpt-tweets-jan-mar-2023
file: chatgpt_tweets.csv
column_map:
  text: content
  user: username
  date: date
  likes: like_count
  retweets: retweet_count
```

For Russian Troll Tweets:

```yaml
source: kaggle
kaggle_slug: vikasg/russian-troll-tweets
file: tweets.csv
column_map:
  text: content
  user: author
  date: publish_date
  region: region
  language: language
  label: account_category
```

If no manifest is provided, columns will be auto-detected.
