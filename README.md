# Vector-Verse

**Semantic Similarity Explorer for Text Datasets**

Vector-Verse is a Python web application that lets you explore and visualize text datasets through the lens of semantic similarity. It embeds your text corpus using OpenAI's embedding models and projects them into an interactive 2D scatter plot using UMAP.

## ✨ Features

- **Multi-Dataset Support**: Poetry, Tweets, Song Lyrics, and custom datasets
- **Semantic Search**: Find similar texts by meaning, not just keywords
- **Hierarchical Zoom**: Lasso-select clusters and re-run UMAP for finer detail
- **Interactive Visualization**: Explore your corpus as a 2D scatter plot
- **Color Dimensions**: Visualize by genre, author, hashtag, user, decade, etc.
- **Multilingual Support**: Works with any language
- **Custom Items**: Add your own texts and see how they relate to the corpus
- **Fast Caching**: First run builds embeddings, subsequent runs load instantly
- **In-App Documentation**: Methodology and Architecture tabs explain how it works

## 🚀 Quick Start

### 1. Clone and Install

```bash
cd vector-verse
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Set Up API Key

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=sk-your-api-key-here
```

Get your API key at: https://platform.openai.com/api-keys

### 3. Add a Dataset

Choose at least one dataset:

#### Poetry (Default)
1. Download from: https://www.kaggle.com/datasets/tgdivy/poetry-foundation-poems
2. Place `PoetryFoundationData.csv` in `data/`

#### Tweets
1. Download from: https://www.kaggle.com/datasets/khalidryder777/500k-chatgpt-tweets-jan-mar-2023
2. Create `data/tweets/` folder
3. Place the CSV inside

#### Lyrics
1. Download from: https://www.kaggle.com/datasets/neisse/scrapped-lyrics-from-6-genres
2. Create `data/lyrics/` folder
3. Place `lyrics-data.csv` (and optionally `artists-data.csv`) inside

### 4. Run the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 📊 Datasets

### Poetry Foundation (~14k poems)
- Source: [Kaggle](https://www.kaggle.com/datasets/tgdivy/poetry-foundation-poems)
- Color by: author, source
- Good for: exploring poetic themes, finding similar writing styles

### Tweets (up to 500k)
- Sources: 
  - [ChatGPT Tweets](https://www.kaggle.com/datasets/khalidryder777/500k-chatgpt-tweets-jan-mar-2023)
  - [Twitter Data](https://www.kaggle.com/datasets/smmmmmmmmmmmm/twitter-data)
- Color by: user, hashtag, date, sentiment label
- Good for: topic clustering, hashtag analysis, sentiment exploration

### Song Lyrics
- Source: [Scrapped Lyrics from 6 Genres](https://www.kaggle.com/datasets/neisse/scrapped-lyrics-from-6-genres)
- Color by: genre, artist, decade, language
- Good for: genre analysis, artist similarity, lyrical theme exploration

## 🔍 Usage

### Browse
Select an item from the sidebar dropdown to see its full text and similar items.

### Search
Enter any text in the search box to find semantically similar items. Your query will be projected onto the visualization as a red star.

### Zoom
1. Use the **lasso tool** (default) to draw around a cluster of points
2. Click **"Zoom Into Selection"** to re-run UMAP on just those items
3. Navigate with **Back** and **Reset** buttons
4. Zoom reveals finer structure that's hidden at the global scale

### Color By
Use the sidebar "Color By" selector to visualize by different dimensions:
- **Genre** (lyrics): See how genres cluster
- **Artist** (lyrics): Compare artists' styles
- **User** (tweets): See which users cluster together
- **Hashtag** (tweets): Visualize topic communities
- **Decade** (lyrics): Explore how music evolved

## 📁 Project Structure

```
vector-verse/
├── app.py                      # Streamlit application
├── config.py                   # Configuration settings
├── requirements.txt            # Python dependencies
├── vector_verse/
│   ├── loaders/                # Dataset loaders (plugin pattern)
│   │   ├── base.py             # BaseDatasetLoader ABC
│   │   ├── poetry.py           # Poetry Foundation loader
│   │   ├── tweets.py           # Tweets CSV loader (NEW)
│   │   ├── lyrics.py           # Lyrics CSV loader (NEW)
│   │   └── custom.py           # Custom items loader
│   ├── embedders/              # Embedding backends
│   │   ├── base.py             # BaseEmbedder ABC
│   │   └── openai_embedder.py  # OpenAI implementation
│   ├── core/
│   │   ├── vector_store.py     # Main orchestrator
│   │   ├── projector.py        # UMAP projection
│   │   └── zoom_manager.py     # Hierarchical zoom (NEW)
│   ├── cache/
│   │   └── manager.py          # Cache persistence
│   └── visualization/
│       └── scatter.py          # Plotly scatter builder
├── data/
│   ├── PoetryFoundationData.csv
│   ├── custom_items.csv
│   ├── tweets/                 # Tweet CSVs go here
│   │   └── README.md
│   └── lyrics/                 # Lyrics CSVs go here
│       └── README.md
└── cache/                      # Auto-generated cache
    ├── poetry_foundation_openai_text-embedding-3-small/
    ├── tweets_openai_text-embedding-3-small/
    └── lyrics_openai_text-embedding-3-small/
```

## ⚙️ Configuration

Edit `config.py` to customize:

```python
# Embedding model
OPENAI_MODEL = "text-embedding-3-small"

# UMAP parameters
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1

# Search results
DEFAULT_K_NEIGHBORS = 10
```

## 🔌 Extending Vector-Verse

### Adding a New Dataset Loader

1. Create `vector_verse/loaders/your_loader.py`:

```python
from .base import BaseDatasetLoader, register_loader

@register_loader("your_dataset")
class YourLoader(BaseDatasetLoader):
    @property
    def name(self) -> str:
        return "your_dataset"
    
    def load(self) -> pd.DataFrame:
        # Return DataFrame with: id, title, author, text, source
        # Plus any metadata columns for color dimensions
        df = pd.read_csv(...)
        return self.validate(df)
    
    def get_color_dimensions(self) -> dict[str, str]:
        return {
            "category": "categorical",
            "year": "sequential",
        }
```

2. Add to `config.py`:

```python
AVAILABLE_DATASETS["your_dataset"] = {
    "loader": "your_dataset",
    "label": "🏷️ Your Dataset",
    "data_check": lambda: (DATA_DIR / "your_data.csv").exists(),
    "color_dimensions": ["category", "year"],
}
```

3. Export in `vector_verse/loaders/__init__.py`:

```python
from .your_loader import YourLoader
```

### Dataset Manifest (Optional)

For explicit column mapping, create `dataset.yaml` in the data folder:

```yaml
source: kaggle
kaggle_slug: username/dataset-name
file: data.csv
column_map:
  text: content
  user: author
  date: timestamp
```

## 💰 Costs

Using OpenAI's text-embedding-3-small:
- ~14k poems: $0.50-1.00 (one-time, cached)
- ~500k tweets: ~$5-10 (one-time, cached)
- Each search query: ~$0.0001

## 🐛 Troubleshooting

### "No datasets found"
Add at least one dataset to the `data/` folder (see Quick Start).

### "OpenAI API key not found"
Create a `.env` file with your API key.

### "Rate limit exceeded"
The app has automatic retry logic. If it persists, wait a minute and try again.

### Zoom feels slow
First zoom into a region is computed fresh. Repeated zooms to the same selection are cached and instant.

### Rebuilding Cache
Use the "🔄 Rebuild Cache" button in the sidebar to re-embed everything.

## 📚 In-App Documentation

The app includes two documentation tabs:

- **📚 Methodology**: Non-technical explanation of embeddings, UMAP, similarity search, and what clusters mean
- **🏗️ Architecture**: Technical diagram of the data flow, caching strategy, and how to extend the system

## 📄 License

MIT License - feel free to use this for your own projects!

## 🙏 Credits

- Poetry dataset: [Kaggle Poetry Foundation Poems](https://www.kaggle.com/datasets/tgdivy/poetry-foundation-poems)
- Tweets dataset: [Kaggle ChatGPT Tweets](https://www.kaggle.com/datasets/khalidryder777/500k-chatgpt-tweets-jan-mar-2023)
- Lyrics dataset: [Kaggle Scrapped Lyrics](https://www.kaggle.com/datasets/neisse/scrapped-lyrics-from-6-genres)
- Embeddings: [OpenAI](https://platform.openai.com/)
- Visualization: [UMAP](https://umap-learn.readthedocs.io/) + [Plotly](https://plotly.com/)
- UI Framework: [Streamlit](https://streamlit.io/)