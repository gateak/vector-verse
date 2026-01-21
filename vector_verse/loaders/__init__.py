"""
Dataset loaders for Vector-Verse.
"""

from .base import BaseDatasetLoader, get_loader, list_loaders, register_loader
from .poetry import PoetryLoader
from .custom import CustomItemsLoader
from .tweets import TweetsCsvLoader
from .lyrics import LyricsCsvLoader
from .combined_tweets import CombinedTweetsLoader

__all__ = [
    "BaseDatasetLoader",
    "get_loader",
    "list_loaders",
    "register_loader",
    "PoetryLoader",
    "CustomItemsLoader",
    "TweetsCsvLoader",
    "LyricsCsvLoader",
    "CombinedTweetsLoader",
]
