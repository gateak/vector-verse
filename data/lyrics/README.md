# Lyrics Dataset

Place lyrics CSV files here. Supported Kaggle datasets:

## Option 1: Scrapped Lyrics from 6 Genres (Multi-file)
- **Source**: https://www.kaggle.com/datasets/neisse/scrapped-lyrics-from-6-genres
- **Download**: Click "Download" on Kaggle, extract
- **Place here**: 
  - `lyrics-data.csv` (required)
  - `artists-data.csv` (optional, for artist names)

## Option 2: MetroLyrics Style (Single file)
- Any CSV with columns like: song, artist, lyrics, genre, year
- Place the CSV file directly in this folder

## Optional: dataset.yaml

For explicit column mapping, create `dataset.yaml`:

```yaml
source: kaggle
kaggle_slug: neisse/scrapped-lyrics-from-6-genres
file: lyrics-data.csv
column_map:
  text: Lyric
  title: SName
  artist_id: ALink
```

If no manifest is provided, columns will be auto-detected.
