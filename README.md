# Vector-Verse

**Semantic Similarity Explorer for Text Datasets**

Vector-Verse is a Python web application that lets you explore and visualize text datasets through the lens of semantic similarity. It embeds your text corpus using OpenAI's embedding models and projects them into an interactive 2D scatter plot using UMAP.

## Features

- **Semantic Search**: Find similar texts by meaning, not just keywords
- **Interactive Visualization**: Explore your corpus as a 2D scatter plot
- **Multilingual Support**: Works with any language (Turkish, English, etc.)
- **Custom Items**: Add your own texts and see how they relate to the corpus
- **Fast Caching**: First run builds embeddings, subsequent runs load instantly
- **Extensible Architecture**: Easy to add new datasets or embedding backends

## Quick Start

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
cp .env.example .env
```

Edit `.env` and add your OpenAI API key:

```
OPENAI_API_KEY=sk-your-api-key-here
```

Get your API key at: https://platform.openai.com/api-keys

### 3. Download the Poetry Dataset

1. Go to: https://www.kaggle.com/datasets/tgdivy/poetry-foundation-poems
2. Download the dataset (you'll need a Kaggle account)
3. Extract `PoetryFoundationData.csv` to the `data/` folder:

```
vector-verse/
└── data/
    └── PoetryFoundationData.csv
```

### 4. Run the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## First Run

On first run, the app will:
1. Load the poetry dataset (~14k poems)
2. Generate embeddings via OpenAI API (~5-10 minutes, ~$0.50-1.00)
3. Compute UMAP projection
4. Save everything to cache

Subsequent runs load from cache in seconds.

## Usage

### Browse
Select an item from the sidebar dropdown to see its full text and similar items.

### Search
Enter any text in the search box to find semantically similar items. Your query will be projected onto the visualization as a red star.

### Visualize
The scatter plot shows all items in 2D space:
- **Blue circles**: Poetry dataset items
- **Orange diamonds**: Your custom items
- **Green highlight**: Selected item
- **Purple highlights**: Similar items
- **Red star**: Search query

## Adding Custom Items

Edit `data/custom_items.csv` to add your own texts:

```csv
title,author,text,language
"My Poem","Your Name","Your poem text here...","en"
"Şiirim","Adınız","Türkçe şiir metni...","tr"
```

Custom items:
- Are always included (never sampled away)
- Appear as orange diamonds in the visualization
- Work with any language

## Project Structure

```
vector-verse/
├── app.py                      # Streamlit application
├── config.py                   # Configuration settings
├── requirements.txt            # Python dependencies
├── .env.example               # API key template
├── vector_verse/
│   ├── loaders/               # Dataset loaders (plugin pattern)
│   │   ├── base.py            # BaseDatasetLoader ABC
│   │   ├── poetry.py          # Poetry Foundation loader
│   │   └── custom.py          # Custom items loader
│   ├── embedders/             # Embedding backends (strategy pattern)
│   │   ├── base.py            # BaseEmbedder ABC
│   │   └── openai_embedder.py # OpenAI implementation
│   ├── core/
│   │   ├── vector_store.py    # Main orchestrator
│   │   └── projector.py       # UMAP projection
│   ├── cache/
│   │   └── manager.py         # Cache persistence
│   └── visualization/
│       └── scatter.py         # Plotly scatter builder
├── data/
│   ├── PoetryFoundationData.csv  # Download from Kaggle
│   └── custom_items.csv          # Your custom items
└── cache/                         # Auto-generated cache
```

## Configuration

Edit `config.py` to customize:

- **Embedding model**: Change `OPENAI_MODEL` (default: text-embedding-3-small)
- **UMAP parameters**: Adjust `UMAP_N_NEIGHBORS`, `UMAP_MIN_DIST`, etc.
- **Search results**: Change `DEFAULT_K_NEIGHBORS`

## Extending Vector-Verse

### Adding a New Dataset

1. Create a new loader in `vector_verse/loaders/`:

```python
from .base import BaseDatasetLoader, register_loader

@register_loader("my_dataset")
class MyDatasetLoader(BaseDatasetLoader):
    @property
    def name(self) -> str:
        return "my_dataset"
    
    def load(self) -> pd.DataFrame:
        # Return DataFrame with: id, title, author, text, source
        ...
```

2. Update `VectorStore` instantiation in `app.py`

### Adding a New Embedding Backend

1. Create a new embedder in `vector_verse/embedders/`:

```python
from .base import BaseEmbedder, register_embedder

@register_embedder("my_embedder")
class MyEmbedder(BaseEmbedder):
    @property
    def name(self) -> str:
        return "my_embedder"
    
    @property
    def dimension(self) -> int:
        return 768  # Your embedding dimension
    
    def embed(self, texts: list[str]) -> np.ndarray:
        # Return normalized embeddings
        ...
```

2. Update `VectorStore` instantiation in `app.py`

## Costs

Using OpenAI's text-embedding-3-small:
- ~14k poems: $0.50-1.00 (one-time, cached)
- Each search query: ~$0.0001

## Troubleshooting

### "Poetry dataset not found"
Download the CSV from Kaggle and place it in `data/PoetryFoundationData.csv`

### "OpenAI API key not found"
Create a `.env` file with your API key (see setup instructions)

### "Rate limit exceeded"
The app has automatic retry logic. If it persists, wait a minute and try again.

### Rebuilding Cache
Check the "Rebuild Cache" button in the sidebar to re-embed everything.

## License

MIT License - feel free to use this for your own projects!

## Credits

- Poetry dataset: [Kaggle Poetry Foundation Poems](https://www.kaggle.com/datasets/tgdivy/poetry-foundation-poems)
- Embeddings: [OpenAI](https://platform.openai.com/)
- Visualization: [UMAP](https://umap-learn.readthedocs.io/) + [Plotly](https://plotly.com/)
