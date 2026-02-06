"""Configuration management for AutoHistorian."""

import os
from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel


class Settings(BaseModel):
    """Application settings."""

    nyt_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    gemini_model: str = "gemini-2.0-flash"
    data_dir: str = "data"


@lru_cache
def get_settings() -> Settings:
    """Load settings from environment variables."""
    load_dotenv()

    return Settings(
        nyt_api_key=os.getenv("AUTOHISTORIAN_NYT_API_KEY"),
        gemini_api_key=os.getenv("AUTOHISTORIAN_GEMINI_API_KEY"),
        gemini_model=os.getenv("AUTOHISTORIAN_GEMINI_MODEL", "gemini-2.0-flash"),
        data_dir=os.getenv("AUTOHISTORIAN_DATA_DIR", "data"),
    )
