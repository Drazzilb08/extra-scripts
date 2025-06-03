from __future__ import annotations
import sys
import os
import re
import csv
import logging
import unicodedata
import time
import argparse
import shutil
from typing import Pattern, Optional, Any, Callable
from collections import defaultdict
from difflib import SequenceMatcher
from types import SimpleNamespace
from dataclasses import dataclass, field
from functools import wraps
import json
from datetime import datetime, timedelta
import signal
import atexit
import sqlite3

version = "1.4.0"

if sys.version_info < (3, 9):
    print("Python 3.9 or higher is required. Detected version: {}.{}.{}".format(*sys.version_info[:3]))
    exit(1)

try:
    from tmdbapis import TMDbAPIs
    from ratelimit import limits, RateLimitException
    from unidecode import unidecode
    from dotenv import load_dotenv
    from rich.console import Console
    from rich.table import Table
    from rich.prompt import Prompt
    from rich.traceback import Traceback
    from rich.progress import (
        Progress,
        TextColumn,
        BarColumn,
        MofNCompleteColumn,
        TaskProgressColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )
except ImportError as e:
    missing = getattr(e, "name", None) or str(e)
    print(
        f"❌ Missing dependency: {missing}. Please install all dependencies with 'pip install -r requirements.txt'."
    )
    exit(1)

try:
    import subprocess

    BUILD_NUMBER = (
        subprocess.check_output(["git", "rev-list", "--count", "HEAD"], stderr=subprocess.DEVNULL)
        .decode()
        .strip()
    )
    FULL_VERSION = f"{version}.build{BUILD_NUMBER}"
except Exception:
    FULL_VERSION = version

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(SCRIPT_DIR, "logs")
CACHE_DIR = os.path.join(SCRIPT_DIR, "cache")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
CACHE_PATH = os.path.join(CACHE_DIR, "idarr_cache.db")
LOG_PATH = os.path.join(LOG_DIR, "idarr.log")
YEAR_REGEX: Pattern = re.compile(r"\s?\((\d{4})\)(?!.*Collection).*")
SEASON_PATTERN: Pattern = re.compile(
    r"(?:\s*-\s*Season\s*\d+|_Season\d{1,2}|\s*-\s*Specials|_Specials)", re.IGNORECASE
)
TMDB_ID_REGEX: Pattern = re.compile(r"tmdb[-_\s](\d+)")
TVDB_ID_REGEX: Pattern = re.compile(r"tvdb[-_\s](\d+)")
IMDB_ID_REGEX: Pattern = re.compile(r"imdb[-_\s](tt\d+)")
TITLE_YEAR_REGEX: Pattern = re.compile(r"^(.*?)(?: \((\d{4})\))?$")
UNMATCHED_CASES: list[dict[str, Any]] = []
TVDB_MISSING_CASES: list[dict[str, Any]] = []
RECLASSIFIED: list[dict[str, Any]] = []
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp"]
PENDING_MATCHES_PATH = os.path.join(LOG_DIR, "pending_matches.jsonc")
IGNORED_TITLES_PATH = os.path.join(LOG_DIR, "ignored_titles.jsonc")
COUNTRY_CODES = {
    "NL",
    "FR",
    "US",
    "UK",
    "CA",
    "DE",
    "ES",
    "IT",
    "JP",
    "RU",
    "KR",
    "BR",
    "PL",
    "SE",
    "NO",
    "DK",
    "FI",
    "CN",
    "IN",
    "AU",
    "NZ",
    "IE",
    "PT",
    "MX",
    "TR",
}
console = Console()
status_context = None
progress_context = None
progress_bar = None


# --- Log Manager class ---
class LogManager:
    """
    Manages logging to file and console with color support, log rotation, and quiet mode.
    Attributes:
        logger: The underlying logging.Logger instance.
        quiet: If True, suppresses console output.
        levels: Mapping of log level names to methods.
    """

    COLORS = {
        "WHITE": "\033[97m",
        "YELLOW": "\033[93m",
        "RED": "\033[91m",
        "BLUE": "\033[94m",
        "GREEN": "\033[92m",
    }

    def __init__(self, name: str, LOG_PATH: str, max_logs: int = 10):
        """
        Initialize the LogManager.
        Args:
            name: Logger name.
            LOG_PATH: Path to log file.
            max_logs: Maximum rotated log files to keep.
        """
        self.logger = logging.getLogger(name)
        self._rotate_logs(LOG_PATH, max_logs=max_logs)
        fh = logging.FileHandler(LOG_PATH, mode="w")
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"))
        self.logger.addHandler(fh)
        self.logger.setLevel(logging.INFO)
        self.quiet = False

        self.levels = {"info": self.info, "warning": self.warning, "debug": self.debug, "error": self.error}

    def _rotate_logs(self, base_LOG_PATH, max_logs=10):
        """
        Rotate old log files, keeping up to max_logs files.
        Side effects: Renames or deletes log files in log directory.
        """
        log_dir = os.path.dirname(base_LOG_PATH)
        for i in range(max_logs, 0, -1):
            src = os.path.join(log_dir, f"idarr.{i-1}.log") if i > 1 else base_LOG_PATH
            dst = os.path.join(log_dir, f"idarr.{i}.log")
            if os.path.exists(src):
                if i == max_logs:
                    try:
                        os.remove(dst)
                    except Exception:
                        pass
                try:
                    os.rename(src, dst)
                except Exception:
                    pass

    def configure(self, *, quiet: bool = False, level: str = "INFO"):
        """
        Configure quiet mode and log level.
        Args:
            quiet: If True, suppress console output.
            level: Logging level as string (e.g. "INFO", "DEBUG").
        """
        self.quiet = quiet
        self.logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    def __call__(self, level: str, msg: str, color: str = "WHITE", console: bool = True):
        """
        Log a message at a given level, optionally colored and to console.
        Args:
            level: Log level as string.
            msg: Message to log.
            color: Console color name.
            console: If True, output to console.
        Raises:
            ValueError if level is unknown.
        """
        # Usage: log("info", "message", "GREEN")
        if level.lower() in self.levels:
            return self.levels[level.lower()](msg, color, console)
        raise ValueError(f"Unknown log level: {level}")

    def info(self, msg, color="WHITE", console=True):
        """
        Log an info-level message, optionally colored and to console.
        """
        if console and not self.quiet:
            print(f"{self.COLORS.get(color.upper(), '')}{msg}\033[0m")
        self.logger.info(msg)

    def warning(self, msg, color="YELLOW", console=True):
        """
        Log a warning-level message, optionally colored and to console.
        """
        if console and not self.quiet:
            print(f"{self.COLORS.get(color.upper(), '')}{msg}\033[0m")
        self.logger.warning(msg)

    def debug(self, msg, color="BLUE", console=False):
        """
        Log a debug-level message, optionally colored and to console.
        """
        self.logger.debug(msg)
        if console and not self.quiet:
            print(f"{self.COLORS.get(color.upper(), '')}{msg}\033[0m")

    def error(self, msg, color="RED", console=True):
        """
        Log an error-level message, optionally colored and to console.
        """
        if console and not self.quiet:
            print(f"{self.COLORS.get(color.upper(), '')}{msg}\033[0m")
        self.logger.error(msg)


# --- sleep_and_notify decorator ---
def sleep_and_notify(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorator that catches RateLimitException, sleeps for the required period, and retries.
    Args:
        func: Callable to wrap.
    Returns:
        Wrapped function, which retries after rate limiting.
    Side effects:
        Sleeps when rate limit is hit, logs warnings.
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        while True:
            try:
                return func(*args, **kwargs)
            except RateLimitException as e:
                log.warning(
                    f"\033[93m[WARNING]\033[0m Rate limit hit, sleeping for {e.period_remaining:.2f} seconds"
                )
                time.sleep(e.period_remaining)

    return wrapper


def normalize_cache_key(title: str) -> str:
    """
    Normalize a string for use as a cache key.
    - Lowercase, collapse spaces, strip whitespace.
    - Remove most punctuation except Unicode "·".
    - Replace all other punctuation with spaces.
    Args:
        title: Input string.
    Returns:
        Normalized string suitable for cache key.
    """
    allowed = "·"
    norm = "".join(c if c.isalnum() or c.isspace() or c in allowed else " " for c in title)
    norm = re.sub(r"\s+", " ", norm).strip().lower()
    return norm


# --- SQLiteCacheManager class ---
class SQLiteCacheManager:
    """
    Manages a SQLite-backed cache for media metadata.
    Attributes:
        db_path: Path to the SQLite database.
        cache: In-memory dict of cache entries.
        no_cache: If True, disables persistence.
    """

    TABLE_SCHEMA = """
    CREATE TABLE IF NOT EXISTS cache (
        key TEXT PRIMARY KEY,
        tmdb_id INTEGER,
        tvdb_id INTEGER,
        imdb_id TEXT,
        title TEXT,
        year INTEGER,
        type TEXT,
        last_checked TEXT,
        tmdb_url TEXT,
        rename_history TEXT,
        original_filenames TEXT,
        current_filenames TEXT,
        status TEXT
    )
    """

    def __init__(self, path: str, source_dir: Optional[str] = None, no_cache: bool = False):
        """
        Initialize the SQLiteCacheManager.
        Args:
            path: Path to database file.
            source_dir: Optional source directory for files.
            no_cache: If True, disables persistence.
        """
        self.db_path = path.replace(".json", ".db")
        self.source_dir = source_dir
        self.no_cache = no_cache
        self.cache: dict[str, Any] = {}
        self._ensure_table()

    def _ensure_table(self):
        """
        Ensure the cache table exists in the SQLite database.
        No-op if no_cache is True.
        """
        if self.no_cache:
            return
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(self.TABLE_SCHEMA)
            conn.commit()

    def load(self) -> dict[str, Any]:
        """
        Load the cache from SQLite into memory.
        Returns:
            Dict mapping cache keys to entry dicts.
        """
        if self.no_cache:
            self.cache = {}
        else:
            with sqlite3.connect(self.db_path) as conn:
                self.cache = {}
                cursor = conn.execute(
                    "SELECT key, tmdb_id, tvdb_id, imdb_id, title, year, type, last_checked, tmdb_url, rename_history, original_filenames, current_filenames, status FROM cache"
                )
                for row in cursor.fetchall():
                    (
                        key,
                        tmdb_id,
                        tvdb_id,
                        imdb_id,
                        title,
                        year,
                        type_,
                        last_checked,
                        tmdb_url,
                        rename_history,
                        original_filenames,
                        current_filenames,
                        status,
                    ) = row
                    item = {
                        "tmdb_id": tmdb_id,
                        "tvdb_id": tvdb_id,
                        "imdb_id": imdb_id,
                        "title": title,
                        "year": year,
                        "type": type_,
                        "last_checked": last_checked,
                        "tmdb_url": tmdb_url,
                        "rename_history": json.loads(rename_history) if rename_history else [],
                        "original_filenames": json.loads(original_filenames) if original_filenames else [],
                        "current_filenames": json.loads(current_filenames) if current_filenames else [],
                        "status": status or "not_found",
                    }
                    self.cache[key] = item
        return self.cache

    def save(self, active_keys: set[str]) -> None:
        """
        Save current cache to SQLite, only including active_keys.
        Deletes obsolete keys, inserts/updates active ones.
        Args:
            active_keys: Set of keys to persist.
        Side effects:
            Writes to database, may sleep briefly.
        """
        if self.no_cache:
            return
        with console.status("[grey50]Saving cache...", spinner="dots"):
            with sqlite3.connect(self.db_path) as conn:
                keys_to_delete = [k for k in self.cache if k not in active_keys]
                if keys_to_delete:
                    conn.executemany("DELETE FROM cache WHERE key = ?", [(k,) for k in keys_to_delete])
                for k in active_keys:
                    v = self.cache[k]
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO cache
                        (key, tmdb_id, tvdb_id, imdb_id, title, year, type, last_checked, tmdb_url, rename_history, original_filenames, current_filenames, status)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            k,
                            v.get("tmdb_id"),
                            v.get("tvdb_id"),
                            v.get("imdb_id"),
                            v.get("title"),
                            v.get("year"),
                            v.get("type"),
                            v.get("last_checked"),
                            v.get("tmdb_url"),
                            json.dumps(v.get("rename_history", [])),
                            json.dumps(v.get("original_filenames", [])),
                            json.dumps(v.get("current_filenames", [])),
                            v.get("status", "not_found"),
                        ),
                    )
                conn.commit()
            time.sleep(1)

    def verify_and_resolve_collision(self, key: str, value: dict[str, Any]) -> None:
        """
        Detect and resolve cache key collisions by comparing fields.
        If fields differ, keeps the newer entry by last_checked date.
        Args:
            key: Cache key.
            value: New entry dict.
        Side effects:
            Updates in-memory cache.
        """
        if key in self.cache:
            existing = self.cache[key]
            check_fields = ["tmdb_id", "tvdb_id", "imdb_id", "title", "year", "type"]
            diffs = {
                f: (existing.get(f), value.get(f)) for f in check_fields if existing.get(f) != value.get(f)
            }
            if diffs:
                old_dt = existing.get("last_checked")
                new_dt = value.get("last_checked")
                keep = value
                if old_dt and new_dt and old_dt > new_dt:
                    keep = existing
                self.cache[key] = keep

            else:
                self.cache[key] = value
        else:
            self.cache[key] = value

    def upsert(self, key: str, value: dict[str, Any], item: Any = None) -> None:
        """
        Update or insert a cache entry, merging with existing data and ensuring required fields.
        Args:
            key: Cache key.
            value: Dict of new values.
            item: Optional MediaItem to fill missing fields.
        Side effects:
            Updates cache, resolves collisions.
        """
        existing = self.cache.get(key, {})
        merged = {**existing, **value}

        merged = ensure_title_year(merged, item)

        tmdb_id = merged.get("tmdb_id")
        typ = merged.get("type")
        if tmdb_id and typ in ("movie", "tv_series", "collection"):
            if typ == "movie":
                merged["tmdb_url"] = f"https://www.themoviedb.org/movie/{tmdb_id}"
            elif typ == "tv_series":
                merged["tmdb_url"] = f"https://www.themoviedb.org/tv/{tmdb_id}"
            elif typ == "collection":
                merged["tmdb_url"] = f"https://www.themoviedb.org/collection/{tmdb_id}"

        if item:
            hist = merged.get("rename_history", [])
            merged["rename_history"] = hist
        if "status" not in value:
            if tmdb_id:
                merged["status"] = "found"
            else:
                merged["status"] = merged.get("status", "not_found")

        self.verify_and_resolve_collision(key, merged)

    def get_cache_key(self, item: Any) -> str:
        """
        Generate a cache key for a MediaItem based on IDs or normalized title/year.
        Args:
            item: MediaItem or object with title, year, and ID attributes.
        Returns:
            String cache key.
        """
        tmdb = getattr(item, "new_tmdb_id", None) or getattr(item, "tmdb_id", None)
        tvdb = getattr(item, "new_tvdb_id", None) or getattr(item, "tvdb_id", None)
        imdb = getattr(item, "new_imdb_id", None) or getattr(item, "imdb_id", None)
        if tmdb or tvdb or imdb:
            return f"{tmdb or 'no-tmdb'}-{tvdb or 'no-tvdb'}-{imdb or 'no-imdb'}"
        orig_title = getattr(item, "original_title", None)
        orig_year = getattr(item, "original_year", None)
        if orig_title is not None:
            normalized_title = normalize_cache_key(orig_title)
        else:
            normalized_title = normalize_cache_key(item.title)
        year_val = orig_year if orig_year is not None else item.year
        return f"{normalized_title}-{year_val or 'noyear'}"

    def delete(self, query: str | int) -> bool:
        """
        Delete cache entries by TMDB ID (int), or by 'Title (Year)' / 'Title' (str).
        Returns True if any entries were deleted, False otherwise.
        """
        found = False

        # If input is int or a digit string, treat as TMDB ID
        if isinstance(query, int) or (isinstance(query, str) and query.isdigit()):
            tmdb_id = int(query)
            for key, entry in list(self.cache.items()):
                if entry.get("tmdb_id") == tmdb_id:
                    self._delete_by_key(key)
                    found = True
            return found

        m = re.match(TITLE_YEAR_REGEX, query.strip())
        if m:
            title = m.group(1).strip()
            year = int(m.group(2)) if m.group(2) else None
            for key, entry in list(self.cache.items()):
                if entry.get("title") == title and (
                    entry.get("year") == year
                    or (year is None and (entry.get("year") is None or entry.get("year") == ""))
                ):
                    self._delete_by_key(key)
                    found = True
        return found

    def _delete_by_key(self, key: str) -> None:
        """Helper to delete from both cache and DB by key."""
        if key in self.cache:
            del self.cache[key]
        if not self.no_cache:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM cache WHERE key = ?", (key,))
                conn.commit()


def ensure_title_year(entry: dict, item: Any = None) -> dict:
    """
    Ensure the entry dict has title, year, and type fields populated.
    If missing, fills from item if provided.
    Args:
        entry: Dict to update.
        item: Optional MediaItem.
    Returns:
        Updated dict with title/year/type.
    """
    if item is not None:
        entry["title"] = (
            getattr(item, "new_title", None) or getattr(item, "title", "") or entry.get("title", "")
        )
        entry["year"] = getattr(item, "new_year", None)
        if entry["year"] is None:
            entry["year"] = getattr(item, "year", None)
        if entry["year"] is None:
            entry["year"] = getattr(item, "original_year", None)
        if entry["year"] is None:
            entry["year"] = entry.get("year", None)
        entry["type"] = getattr(item, "type", None) or entry.get("type", None)
    else:
        entry["title"] = entry.get("title", "")
        entry["year"] = entry.get("year", None)
        entry["type"] = entry.get("type", None)
    return entry


# --- TMDBQueryService class ---
class TMDBQueryService:
    """
    Service for querying TMDB for media metadata with fuzzy and fallback logic.
    Attributes:
        client: TMDbAPIs client.
        config: IdarrConfig instance.
    """

    def __init__(self, client: TMDbAPIs, config: "IdarrConfig"):
        """
        Initialize TMDBQueryService.
        Args:
            client: TMDbAPIs client.
            config: IdarrConfig instance.
        """
        self.client = client
        self.config = config

    @sleep_and_notify
    @limits(calls=38, period=10)
    def query(
        self,
        search: MediaItem,
        media_type: str,
        retry: bool = False,
        tried: Optional[set] = None,
    ) -> Optional[Any]:
        """
        Query TMDB for a MediaItem, using fuzzy and fallback logic.
        Args:
            search: MediaItem to search for.
            media_type: "movie", "tv_series", or "collection".
            retry: If True, this is a retry/fallback.
            tried: Set of already-tried (title, year, type) tuples.
        Returns:
            TMDB result object or None if not found/ambiguous.
        Side effects:
            Logs to console, sets match_reason/match_failed on search.
        """

        # Immediately return None if match_reason is 'ambiguous'
        if getattr(search, "match_reason", None) == "ambiguous":
            search.match_failed = True
            return None
        if tried is None:
            tried = set()
        key = (search.title, search.year, media_type)
        is_new_attempt = key not in tried
        if not is_new_attempt:
            return None
        tried.add(key)
        self._log_search_attempt(search.title, search.year, media_type, is_new_attempt)

        try:
            # === Step 1: Direct TMDB ID Lookup ===
            if getattr(search, "match_reason", None) == "ambiguous":
                search.match_failed = True
                return None
            if getattr(search, "tmdb_id", None):
                result, reason = self._try_id_lookup(search, media_type)
                if result:
                    self._log_result(
                        f"🎯 TMDB ID {reason} match:",
                        f"  → {getattr(result, 'title', getattr(result, 'name', ''))} ({search.year}) [{getattr(result, 'id', None)}] [{media_type}]",
                    )
                    search.match_reason = reason
                    return result

                # If TMDB ID lookup fails, handle potential TMDB removal for previously matched item
                if (
                    not result
                    and getattr(search, "tmdb_id", None)
                    and getattr(search, "match_reason", None) != "ambiguous"
                ):
                    # Only perform if this item had previously matched and is now missing
                    log.warning(
                        f"❌ TMDB entry for '{search.title}' ({search.year}) [tmdb-{search.tmdb_id}] was deleted from TMDB. Marking as tmdb_removed."
                    )
                    # Build unmatched-style cache key (normalize as per unmatched)
                    unmatched_key = self.config.cache_manager.get_cache_key(
                        type(
                            "Dummy",
                            (),
                            {
                                "title": search.title,
                                "year": search.year,
                                "tmdb_id": None,
                                "tvdb_id": None,
                                "imdb_id": None,
                            },
                        )()
                    )
                    # Remove all tags from all filenames (physical files)
                    search.current_filenames = strip_tags_and_rename(
                        search.current_filenames, dry_run=self.config.dry_run, log=log
                    )
                    # Update entry: clear IDs, set status
                    search.tmdb_id = None
                    search.tvdb_id = None
                    search.imdb_id = None
                    search.match_reason = "tmdb_removed"
                    if hasattr(search, "status"):
                        search.status = "tmdb_removed"
                    # Upsert in cache using new unmatched key
                    self.config.cache_manager.upsert(unmatched_key, search.__dict__)
                    # Return None as no match was found and we handled cleanup
                    search.match_failed = True
                    return None

            # === Step 2: Main TMDB Search ===
            if getattr(search, "match_reason", None) == "ambiguous":
                search.match_failed = True
                return None
            search_results = self._perform_tmdb_search(search, media_type) or []
            if self.config.log_level == "DEBUG" and search_results:
                log.debug(f"[DEBUG] Raw search results for “{search.title}” [{media_type}]:")
                for idx, res in enumerate(search_results, start=1):
                    title = getattr(
                        res,
                        "title",
                        getattr(res, "name", getattr(res, "original_title", "")),
                    )
                    date = getattr(res, "release_date", getattr(res, "first_air_date", None))
                    year_val = (
                        date.year
                        if hasattr(date, "year")
                        else (date[:4] if isinstance(date, str) and len(date) >= 4 else "None")
                    )
                    log.info(f"  {idx}. id={getattr(res,'id',None)}, title=\"{title}\", year={year_val}")

            result, reason = self._try_main_match(search_results, search, media_type)
            if reason == "ambiguous":
                log.warning(
                    f"🤷 Ambiguous result for “{search.title}” ({search.year}) [{media_type}] — Multiple possible matches found."
                )
                search.match_failed = True
                search.match_reason = search.match_reason or "ambiguous"
                return None
            if result:
                year_val = search.year
                if hasattr(result, "release_date") or hasattr(result, "first_air_date"):
                    date_val = getattr(result, "release_date", getattr(result, "first_air_date", None))
                    if isinstance(date_val, str) and date_val[:4].isdigit():
                        year_val = int(date_val[:4])
                    elif hasattr(date_val, "year"):
                        year_val = date_val.year
                self._log_result(
                    f"🎯 {reason} match:",
                    f"  → {getattr(result, 'title', getattr(result, 'name', ''))} ({year_val}) [{getattr(result, 'id', None)}] [{media_type}]",
                )
                search.match_reason = reason
                return result

            # === Step 4: Transformations ===
            if not retry and search_results:
                if getattr(search, "match_reason", None) == "ambiguous":
                    search.match_failed = True
                    return None
                result, reason = self._try_transformations(search_results, search, media_type)
                if result:
                    year_val = search.year
                    if hasattr(result, "release_date") or hasattr(result, "first_air_date"):
                        date_val = getattr(result, "release_date", getattr(result, "first_air_date", None))
                        if isinstance(date_val, str) and date_val[:4].isdigit():
                            year_val = int(date_val[:4])
                        elif hasattr(date_val, "year"):
                            year_val = date_val.year
                    self._log_result(
                        f"🎯 {reason} match:",
                        f"  → {getattr(result, 'title', getattr(result, 'name', ''))} ({year_val}) [{getattr(result, 'id', None)}] [{media_type}]",
                    )
                    search.match_reason = reason
                    return result

            # === Step 3: Fallbacks ===
            if media_type == "movie":
                # Only fallback: try as tv_series with original year
                log.info(f"🔄 Retrying as TV series: “{search.title}”", "YELLOW")
                if getattr(search, "match_reason", None) == "ambiguous":
                    search.match_failed = True
                    return None
                tv_series_result = self.query(search, "tv_series", retry=False, tried=tried)
                if tv_series_result:
                    search.type = "tv_series"
                    return tv_series_result
            # For tv_series, do NOT fallback to year=None
            log.warning(f"🤷 No confident match found for “{search.title}” ({search.year})")
            search.match_failed = True
            if not getattr(search, "match_reason", None):
                search.match_reason = "no_match"
            return None

        except Exception as e:
            # Guard: if ambiguous was set, never trigger fallback logic
            if getattr(search, "match_reason", None) == "ambiguous":
                search.match_failed = True
                search.match_reason = search.match_reason or "ambiguous"
                # Safety: never retry fallbacks if ambiguous found
                return None
            log.warning(
                f"[WARNING] Failed to query TMDB for '{search.title}' ({search.year}) as {media_type}: {e}"
            )

            # Only fallback on "No Results Found"
            if "No Results Found" in str(e):
                if getattr(search, "match_reason", None) == "ambiguous":
                    search.match_failed = True
                    return None
                result, reason = self._try_transformations([], search, media_type)
                if result:
                    self._log_result(
                        f"🎯 {reason} match:",
                        f"  → {getattr(result, 'title', getattr(result, 'name', ''))} ({search.year}) [{getattr(result, 'id', None)}] [{media_type}]",
                    )
                    search.match_reason = reason
                    return result

                # Exception fallback logic: only fallback to tv_series with year
                if media_type == "movie":
                    log.info(f"🔄 Retrying as TV series: “{search.title}”", "YELLOW")
                    if getattr(search, "match_reason", None) == "ambiguous":
                        search.match_failed = True
                        return None
                    tv_series_result = self.query(search, "tv_series", retry=False, tried=tried)
                    if tv_series_result:
                        search.type = "tv_series"
                        return tv_series_result

    def _fetch_by_tmdb_id(self, search: MediaItem, media_type: str) -> Optional[Any]:
        """
        Fetch TMDB details for the given ID, checking all possible types.
        Args:
            search: MediaItem to search for.
            media_type: Media type string.
        Returns:
            Result object if found and matches, else None.
        Side effects:
            Sets search.match_reason/match_failed.
        """
        try_types = ["movie", "tv_series", "collection"]
        if media_type in try_types:
            try_types.remove(media_type)
        try_types.insert(0, media_type)
        tmdb_id = int(search.tmdb_id) if search.tmdb_id is not None else None

        for m_type in try_types:
            try:
                result = None
                if m_type == "movie":
                    result = self.client._api.movies_get_details(tmdb_id)
                elif m_type == "tv_series":
                    # Fetch main show object
                    show = self.client._api.tv_get_details(tmdb_id)
                    # Hydrate with external_ids
                    if show and isinstance(show, dict):
                        external_ids = self.client._api.tv_get_external_ids(tmdb_id)
                        if external_ids:
                            show["imdb_id"] = external_ids.get("imdb_id")
                            show["tvdb_id"] = external_ids.get("tvdb_id")
                    result = show
                elif m_type == "collection":
                    result = self.client._api.collections_get_details(tmdb_id)

                if result and isinstance(result, dict):
                    result = SimpleNamespace(**result)

                if result:
                    res_id = getattr(result, "id", None)
                    if res_id == tmdb_id:
                        # Check for exact title/year match
                        detail_title = getattr(
                            result,
                            "title",
                            getattr(result, "name", getattr(result, "original_title", "")),
                        )
                        detail_year = None
                        date = getattr(result, "release_date", None) or getattr(
                            result, "first_air_date", None
                        )
                        if isinstance(date, str) and len(date) >= 4 and date[:4].isdigit():
                            detail_year = int(date[:4])
                        elif hasattr(date, "year"):
                            detail_year = date.year
                        # Compare normalized title/year
                        if normalize_with_aliases(detail_title) == normalize_with_aliases(search.title) and (
                            search.year is None or detail_year == search.year
                        ):
                            search.match_reason = "id_exact"
                            result.media_type = m_type
                            return result

                        # Otherwise, fuzzy check
                        candidates = self._fuzzy_match_candidates(
                            [result],
                            search,
                            strict=False,
                            ratio_threshold=0.80,
                            jaccard_threshold=0.60,
                            year_tolerance=2,
                        )
                        if candidates:
                            search.match_reason = "id_fuzzy"
                            result.media_type = m_type
                            return result
                        # Otherwise, mismatch
                        search.match_failed = True
                        search.match_reason = "id data mismatch"
                        return None
            except Exception:
                continue
        return None

    def rehydrate_missing_tvdb_ids(self, cache, max_age_days=7):
        """
        For all TV series in the cache missing a tvdb_id and not checked recently,
        attempt to rehydrate from TMDB.
        Side effect: Updates cache entries in-place.
        """
        now = datetime.now()
        updated = 0
        stale_delta = timedelta(days=max_age_days)

        for cache_key, entry in cache.items():
            if entry.get("type") == "tv_series" and not entry.get("tvdb_id"):
                last_checked = entry.get("last_checked")
                stale = True
                if last_checked:
                    try:
                        checked_dt = (
                            datetime.strptime(last_checked, "%Y-%m-%d %H:%M:%S")
                            if isinstance(last_checked, str)
                            else last_checked
                        )
                        if now - checked_dt < stale_delta:
                            stale = False
                    except Exception:
                        pass
                if not stale:
                    continue  # Skip if checked recently

                tmdb_id = entry.get("tmdb_id")
                if tmdb_id:
                    log.debug(
                        f"🔄 Rehydrating TV series missing tvdb_id (stale): {entry.get('title')} [{tmdb_id}]"
                    )
                    try:
                        result = self._fetch_by_tmdb_id(
                            search=type(
                                "S",
                                (),
                                {
                                    "tmdb_id": tmdb_id,
                                    "title": entry.get("title"),
                                    "year": entry.get("year"),
                                },
                            )(),
                            media_type="tv_series",
                        )
                        if result:
                            entry["tvdb_id"] = getattr(result, "tvdb_id", None)
                            entry["imdb_id"] = getattr(result, "imdb_id", None)
                            entry["last_checked"] = now.strftime("%Y-%m-%d %H:%M:%S")
                            updated += 1
                    except Exception as e:
                        log.warning(f"Failed to rehydrate {entry.get('title')} [{tmdb_id}]: {e}")
        if updated != 0:
            log.info(f"✅ Rehydrated {updated} TV series entries missing tvdb_id (stale only).")
            return True
        return False

    def _fuzzy_match_candidates(
        self,
        search_results: list[Any],
        search: MediaItem,
        *,
        strict: bool = True,
        ratio_threshold: float = 0.90,
        jaccard_threshold: float = 0.85,
        year_tolerance: int = 0,
    ) -> Any:
        """
        Perform fuzzy matching of search results against a MediaItem.
        Args:
            search_results: List of TMDB result objects.
            search: MediaItem to match.
            strict: If True, only accept high-confidence matches.
            ratio_threshold: Minimum SequenceMatcher ratio.
            jaccard_threshold: Minimum word Jaccard similarity.
            year_tolerance: Allowed year difference.
        Returns:
            If strict, returns single candidate or None.
            If not strict, returns list of candidates sorted by score.
        """

        def word_jaccard(a: str, b: str) -> float:
            words_a = set(re.findall(r"\w+", a.lower()))
            words_b = set(re.findall(r"\w+", b.lower()))
            if not words_a or not words_b:
                return 0.0
            intersection = words_a & words_b
            union = words_a | words_b
            return len(intersection) / len(union)

        norm_search = normalize_with_aliases(search.title)
        candidates = []
        scored = []
        for res in search_results:
            title = getattr(res, "title", getattr(res, "name", ""))
            norm_res = normalize_with_aliases(title)
            ratio = SequenceMatcher(None, norm_search, norm_res).ratio()
            date = getattr(res, "release_date", getattr(res, "first_air_date", ""))
            res_year = None
            if isinstance(date, str) and date[:4].isdigit():
                res_year = int(date[:4])
            elif hasattr(date, "year"):
                res_year = date.year
            jaccard = word_jaccard(norm_search, norm_res)
            y_score = (
                1.0
                if res_year == search.year
                else 0.5 if res_year and search.year and abs(res_year - search.year) <= 1 else 0
            )
            score = ratio * 2 + y_score
            scored.append((score, res))
            if strict:
                year_ok = (search.year is None and res_year is None) or (
                    res_year is not None
                    and search.year is not None
                    and abs(res_year - search.year) <= year_tolerance
                )
                if ratio >= ratio_threshold and jaccard >= jaccard_threshold and year_ok:
                    candidates.append(res)
                elif 0.0 <= ratio_threshold - ratio <= 0.05:
                    log.debug(
                        f"📉 Near-match: '{title}' (Ratio: {ratio:.3f}, Jaccard: {jaccard:.3f}, Year: {res_year})"
                    )
            else:
                if score > 1.0:
                    candidates.append(res)
        if strict:
            return candidates[0] if len(candidates) == 1 else None
        else:
            scored.sort(key=lambda x: x[0], reverse=True)
            return candidates

    def _perform_tmdb_search(self, search: MediaItem, media_type: str) -> Optional[list[Any]]:
        """
        Perform a TMDB search for the given MediaItem and media type.
        Args:
            search: MediaItem to search for.
            media_type: Media type string.
        Returns:
            List of TMDB result objects or None.
        """
        if media_type == "collection":
            return self.client.collection_search(query=search.title)
        elif media_type == "movie":
            return self.client.movie_search(query=search.title, year=search.year)
        elif media_type == "tv_series":
            return self.client.tv_search(query=search.title, first_air_date_year=search.year)
        else:
            log.info(f"[SKIPPED] Unsupported media type '{media_type}' for '{search.title}'")
            return None

    def _match_by_id(self, search_results: list[Any], search: MediaItem, media_type: str) -> Optional[Any]:
        """
        Return first result matching TMDB, TVDB, or IMDB ID.
        """
        for res in search_results:
            if (
                (getattr(search, "tmdb_id", None) and getattr(res, "id", None) == search.tmdb_id)
                or (getattr(search, "tvdb_id", None) and getattr(res, "tvdb_id", None) == search.tvdb_id)
                or (getattr(search, "imdb_id", None) and getattr(res, "imdb_id", None) == search.imdb_id)
            ):
                return res
        return None

    def _exact_match_shortcut(self, search_results: list[Any], search: MediaItem) -> Optional[Any]:
        """
        Return first result whose normalized title and year exactly match the search MediaItem.
        """
        norm_search = normalize_with_aliases(search.title)
        for res in search_results:
            title = getattr(res, "title", getattr(res, "name", ""))
            if normalize_with_aliases(title) == norm_search:
                date = getattr(res, "release_date", getattr(res, "first_air_date", ""))
                year = (
                    date.year
                    if hasattr(date, "year")
                    else (int(date[:4]) if isinstance(date, str) and date[:4].isdigit() else None)
                )
                if year == search.year:
                    return res
        return None

    def _alternate_titles_fallback(
        self, search_results: list[Any], search: MediaItem, media_type: str
    ) -> Optional[Any]:
        """
        Search for a match in alternative titles of results.
        """
        norm_search = normalize_with_aliases(search.title)
        for res in search_results:
            alt_list = getattr(res, "alternative_titles", [])
            for alt in alt_list:
                cand = alt.get("title") if isinstance(alt, dict) else str(alt)
                if normalize_with_aliases(cand) == norm_search:
                    return res
        return None

    def _match_by_original_title(
        self, search_results: list[Any], search: MediaItem, media_type: str
    ) -> Optional[Any]:
        """
        Return first result whose original_title matches the search title (normalized).
        """
        for res in search_results:
            orig_title = getattr(res, "original_title", None)
            if orig_title and normalize_with_aliases(orig_title) == normalize_with_aliases(search.title):
                return res
        return None

    def _log_search_attempt(self, title, year, media_type, is_new_attempt):
        """
        Log the start of a TMDB search if this is a new attempt.
        """
        if is_new_attempt:
            log.info(f"🔍 Searching TMDB for “{title}” ({year}) [{media_type}]...")

    def _log_result(self, header, msg, color="GREEN"):
        """
        Log a result header and message with optional color.
        """
        log.info(header)
        log.info(msg, color)

    def _try_id_lookup(self, search, media_type):
        """
        Try to fetch by TMDB ID; returns (result, reason) or (None, None).
        """
        result = self._fetch_by_tmdb_id(search, media_type)
        if result:
            reason = search.match_reason or "id"
            return result, reason
        return None, None

    def _try_main_match(self, search_results, search, media_type):
        """
        Try to match using all main strategies (ID, ambiguous, exact, original, alternate, fuzzy, etc).
        Returns (result, reason string) or (None, None).
        """
        # 1. ID match
        id_match = self._match_by_id(search_results, search, media_type)
        if id_match:
            return id_match, "id"
        # 2. Ambiguous check (normalize titles)
        norm_search = normalize_with_aliases(search.title)
        same_title_year = []
        for res in search_results:
            res_title = getattr(res, "title", getattr(res, "name", ""))
            norm_res_title = normalize_with_aliases(res_title)
            date = getattr(res, "release_date", getattr(res, "first_air_date", ""))
            res_year = (
                date.year
                if hasattr(date, "year")
                else (int(date[:4]) if isinstance(date, str) and date[:4].isdigit() else None)
            )
            if norm_res_title == norm_search and res_year == search.year:
                same_title_year.append(res)
        if len(same_title_year) > 1:
            search.match_failed = True
            search.match_reason = "ambiguous"
            return None, "ambiguous"
        # 3. Exact shortcut
        shortcut = self._exact_match_shortcut(search_results, search)
        if shortcut:
            return shortcut, "exact"
        # 4. Original title
        orig_match = self._match_by_original_title(search_results, search, media_type)
        if orig_match:
            return orig_match, "original"
        # 5. Alternate-title
        alt = self._alternate_titles_fallback(search_results, search, media_type)
        if alt:
            return alt, "alternate"
        # 6. Fuzzy
        fuzzy = self._fuzzy_match_candidates(search_results, search, strict=True)
        if fuzzy:
            return fuzzy, "fuzzy_norm"
        # 6. Fuzzy - wider year diff if search_results is 1 item
        if len(search_results) == 1:
            fuzzy = self._fuzzy_match_candidates(
                search_results,
                search,
                strict=True,
                ratio_threshold=0.9,
                jaccard_threshold=0.85,
                year_tolerance=2,
            )
            if fuzzy:
                return fuzzy, "fuzzy_year_diff"
        # 7. Fuzzy alternate
        alt_candidates = []
        for res in search_results:
            alt_list = getattr(res, "alternative_titles", [])
            for alt in alt_list:
                alt_title = alt.get("title") if isinstance(alt, dict) else str(alt)
                alt_candidates.append(SimpleNamespace(title=alt_title, _orig_res=res))
        fuzzy_alt = None
        if alt_candidates:
            fuzzy_alt_result = self._fuzzy_match_candidates(alt_candidates, search, strict=True)
            if fuzzy_alt_result:
                fuzzy_alt = getattr(fuzzy_alt_result, "_orig_res", None)
        if fuzzy_alt:
            return fuzzy_alt, "fuzzy_alternate"
        return None, None

    def _try_transformations(self, search_results, search, media_type):
        """
        Try various title transformations and search again for a match.
        Returns (result, reason string) or (None, None).
        """
        transformations = [
            (
                lambda s: re.sub(rf"\s*\((?:{'|'.join(COUNTRY_CODES)})\)", "", s, flags=re.IGNORECASE),
                lambda s: bool(re.search(rf"\((?:{'|'.join(COUNTRY_CODES)})\)", s, flags=re.IGNORECASE)),
            ),
            (lambda s: unidecode(s), lambda s: any(ord(c) > 127 for c in s)),
            (lambda s: s.replace("_", " "), lambda s: "_" in s),
            (lambda s: s.replace("-", " "), lambda s: "-" in s),
            (lambda s: s.replace("_", ":"), lambda s: "_" in s),
            (lambda s: s.replace("-", ":"), lambda s: "-" in s),
            (lambda s: s.replace("-", "\\"), lambda s: "-" in s),
            (lambda s: s.replace("+", "\\"), lambda s: "+" in s),
        ]
        for transform, predicate in transformations:
            if predicate(search.title):
                alt_title = transform(search.title)
                if alt_title != search.title:
                    temp_search = MediaItem(
                        config=search.config,
                        type=search.type,
                        title=alt_title,
                        year=search.year,
                        tmdb_id=search.tmdb_id,
                        tvdb_id=search.tvdb_id,
                        imdb_id=search.imdb_id,
                        files=search.files,
                    )
                    try:
                        if not search_results:
                            alt_results = self._perform_tmdb_search(temp_search, media_type) or []
                        else:
                            alt_results = search_results
                    except Exception as e:
                        log.warning(f"🔁 Transformation search failed for '{alt_title}': {e}")
                        continue
                    shortcut = self._exact_match_shortcut(alt_results, temp_search)
                    if shortcut:
                        log.info(f"🔁 Retrying TMDB search with transformed title: '{alt_title}'", "BLUE")
                        return shortcut, "exact_transform"
                    orig_match = self._match_by_original_title(alt_results, temp_search, media_type)
                    if orig_match:
                        log.info(f"🔁 Retrying TMDB search with transformed title: '{alt_title}'", "BLUE")
                        return orig_match, "original_transform"
                    alt = self._alternate_titles_fallback(alt_results, temp_search, media_type)
                    if alt:
                        log.info(f"🔁 Retrying TMDB search with transformed title: '{alt_title}'", "BLUE")
                        return alt, "alternate_transform"
                    fuzzy = self._fuzzy_match_candidates(alt_results, temp_search, strict=True)
                    if fuzzy:
                        log.info(f"🔁 Retrying TMDB search with transformed title: '{alt_title}'", "BLUE")
                        return fuzzy, "fuzzy_transform"
                    alt_candidates = []
                    for res in alt_results:
                        alt_list = getattr(res, "alternative_titles", [])
                        for alt in alt_list:
                            alt_title2 = alt.get("title") if isinstance(alt, dict) else str(alt)
                            alt_candidates.append(SimpleNamespace(title=alt_title2, _orig_res=res))
                    if alt_candidates:
                        fuzzy_alt_result = self._fuzzy_match_candidates(
                            alt_candidates, temp_search, strict=True
                        )
                        if fuzzy_alt_result:
                            fuzzy_alt = getattr(fuzzy_alt_result, "_orig_res", None)
                            if fuzzy_alt:
                                log.info(
                                    f"🔁 Retrying TMDB search with transformed title: '{alt_title}'",
                                    "BLUE",
                                )
                                return fuzzy_alt, "fuzzy_alternate_transform"

        return None, None


# --- IdarrConfig dataclass ---
@dataclass
class IdarrConfig:
    """
    Configuration for idarr processing, including cache, TMDB API, and user options.
    Attributes:
        dry_run: If True, don't modify files.
        quiet: If True, suppress most output.
        log_level: Logging level.
        source_dir: Directory to scan.
        tmdb_api_key: TMDB API key.
        frequency_days: Cache expiration (days).
        tvdb_frequency: Days before retry TVDb.
        cache_path: Path to cache file.
        no_cache: If True, disables cache.
        clear_cache: If True, clears cache at start.
        remove_non_image_files: If True, deletes non-image files.
        limit: Limit number of items.
        filter: If True, filter items.
        type: Media type filter.
        year: Year filter.
        contains: String filter.
        id: ID filter.
        show_unmatched: If True, show unmatched items.
        revert: If True, revert renames.
        skip_collections: If True, skip collections.
        ignore_file: Path to ignore file.
        pending_matches: Dict of pending matches.
        ignored_title_keys: Set of ignored title keys.
        cache_manager: SQLiteCacheManager instance.
        cache: Dict of cache entries.
        tmdb_query_service: TMDBQueryService instance.
        _api_calls: Counter for API calls.
    """

    dry_run: bool = False
    quiet: bool = False
    log_level: str = "INFO"
    source_dir: Optional[str] = None
    tmdb_api_key: Optional[str] = None
    frequency_days: int = field(default_factory=lambda: int(os.environ.get("FREQUENCY_DAYS", "30")))
    tvdb_frequency: int = field(default_factory=lambda: int(os.environ.get("TVDB_FREQUENCY", "7")))
    cache_path: str = field(default_factory=lambda: os.path.join(SCRIPT_DIR, "cache", "idarr_cache.json"))
    no_cache: bool = False
    clear_cache: bool = False
    remove_non_image_files: bool = False
    limit: Optional[int] = None
    filter: bool = False
    type: Optional[str] = None
    year: Optional[int] = None
    contains: Optional[str] = None
    id: Optional[str] = None
    show_unmatched: bool = False
    revert: bool = False
    skip_collections: bool = False
    ignore_file: Optional[str] = None
    pending_matches: dict[str, str] = field(default_factory=dict)
    ignored_title_keys: set[str] = field(default_factory=set)
    cache_manager: "SQLiteCacheManager" = field(init=False)
    cache: dict[str, Any] = field(init=False)
    tmdb_query_service: "TMDBQueryService" = field(init=False)
    _api_calls: int = 0


def save_pending_matches(pending_matches: dict, pending_file: str = PENDING_MATCHES_PATH):
    """
    Write pending matches dict to pending_matches.jsonc with standard header.
    """
    try:
        with open(pending_file, "w", encoding="utf-8") as f:
            f.write('// List of pending matches in the form "Title (Year)": "add_tmdb_url_here",\n')
            f.write(
                '// Replace "add_tmdb_url_here" with a TMDB URL, ID, or use "ignore" to send to ignored_titles.jsonc.\n'
            )
            f.write("// Example:\n")
            f.write('// "Some Movie (2023)": "https://www.themoviedb.org/movie/12345"\n')
            json.dump(dict(sorted(pending_matches.items())), f, indent=2, ensure_ascii=False)  # Sort by key
            f.write("\n")
    except Exception as e:
        log.warning(f"⚠️ Failed to save updated pending matches: {e}")


def update_pending_matches_from_cache(config) -> dict[str, str]:
    """
    Build pending matches from cache: items with status == 'not_found' or status == 'tmdb_removed'.
    Args:
        config: IdarrConfig or similar with .cache attribute.
    Returns:
        Dict mapping title keys to placeholder string.
    """
    pending = {}
    cache = getattr(config, "cache", {})
    for entry in cache.values():
        status = entry.get("status", "not_found")
        if status not in ("not_found", "tmdb_removed"):
            continue
        title = entry.get("title", "")
        year = entry.get("year")
        if title and year:
            key = f"{title} ({year})"
        else:
            key = title
        pending[key] = "add_tmdb_url_here"
    return pending


def is_recent(last_checked: str, config: "IdarrConfig") -> bool:
    """
    Return True if last_checked is within config.frequency_days from now.
    Args:
        last_checked: ISO date string.
        config: IdarrConfig with frequency_days.
    Returns:
        True if recent, False otherwise.
    """
    freq_days = config.frequency_days
    try:
        last_time = datetime.fromisoformat(last_checked)
        return datetime.now() - last_time < timedelta(days=freq_days)
    except Exception:
        return False


# --- MediaItem class ---
class MediaItem:
    """
    Represents a media item (movie, tv_series, or collection) with associated metadata and files.
    Attributes:
        type: Media type ("movie", "tv_series", "collection").
        title: Canonical title.
        year: Year of release.
        tmdb_id, tvdb_id, imdb_id: Source IDs.
        files: List of associated file paths.
        new_title, new_year, new_tmdb_id, ...: Updated values after enrichment.
        config: Reference to IdarrConfig or SimpleNamespace.
        original_title, original_year: Initial values.
        match_failed, match_reason, unmatched, reclassified, is_ambiguous: Status flags.
        renamed: True if a rename was performed.
    """

    def __init__(
        self,
        config: "IdarrConfig",
        type: str,
        title: str,
        year: Optional[int],
        tmdb_id: Optional[int],
        tvdb_id: Optional[int] = None,
        imdb_id: Optional[str] = None,
        files: Optional[list[str]] = None,
    ) -> None:
        self.type: str = type
        self.title: str = title
        self.year: Optional[int] = year
        self.tmdb_id: Optional[int] = tmdb_id
        self.tvdb_id: Optional[int] = tvdb_id
        self.imdb_id: Optional[str] = imdb_id
        self.files: list[str] = files or []
        self.new_title: Optional[str] = None
        self.new_year: Optional[int] = None
        self.new_tmdb_id: Optional[int] = None
        self.new_tvdb_id: Optional[int] = None
        self.new_imdb_id: Optional[str] = None
        self.match_failed: bool = False
        self.match_reason: Optional[str] = None
        self.unmatched: bool = False
        self.reclassified: bool = False
        self.is_ambiguous: bool = False
        self.config: SimpleNamespace = config
        self.renamed: bool = False
        self.original_title: str = title
        self.original_year: Optional[int] = year
    
    def __repr__(self):
        files = getattr(self, 'files', [])
        file_list = [os.path.basename(f) for f in files]
        return f"<MediaItem title={self.title!r}, year={self.year!r}, files={file_list}>"

    def _update_and_save_cache(self, key: str, value: dict[str, Any]) -> None:
        """
        Update the cache entry for this media item.
        - Stores all filenames as lists in 'original_filenames' and 'current_filenames'.
        - Removes obsolete singular filename keys.
        Args:
            key: Cache key.
            value: Entry dict to store.
        Side effects:
            Updates cache and filenames, may update tmdb_url.
        """
        v = value.copy()
        v["title"] = getattr(self, "title", "") or getattr(self, "original_title", "") or v.get("title", "")
        year_val = getattr(self, "year", None)
        if year_val is None:
            year_val = getattr(self, "original_year", None)
        v["year"] = year_val if year_val is not None else v.get("year", None)

        tmdb_id = v.get("tmdb_id")
        typ = v.get("type")
        tmdb_url = None
        if tmdb_id and typ in ("movie", "tv_series", "collection"):
            if typ == "movie":
                tmdb_url = f"https://www.themoviedb.org/movie/{tmdb_id}"
            elif typ == "tv_series":
                tmdb_url = f"https://www.themoviedb.org/tv/{tmdb_id}"
            elif typ == "collection":
                tmdb_url = f"https://www.themoviedb.org/collection/{tmdb_id}"
        if tmdb_url:
            v["tmdb_url"] = tmdb_url

        hist = v.get("rename_history", [])
        originals_from_history = {h["from"] for h in hist if "from" in h}
        originals_from_files = {os.path.basename(f) for f in (self.files or [])}
        all_originals = originals_from_history | originals_from_files
        v["original_filenames"] = sorted(all_originals)

        source_dir = getattr(self.config, "source_dir", None)
        real_files = set(os.listdir(source_dir)) if source_dir and os.path.isdir(source_dir) else set()
        all_possible_filenames = set(v["original_filenames"])
        all_possible_filenames.update([h["to"] for h in hist if "to" in h])
        current_files = sorted({f for f in all_possible_filenames if f in real_files})
        v["current_filenames"] = current_files

        v.pop("original_filename", None)
        v.pop("current_filename", None)
        if "status" not in v:
            if v.get("tmdb_id"):
                v["status"] = "found"
            elif v.get("no_result"):
                v["status"] = "not_found"
            else:
                v["status"] = "not_found"
        self.config.cache_manager.verify_and_resolve_collision(key, v)
        self.config.cache = self.config.cache_manager.cache

    def enrich(self) -> bool:
        """
        Enrich this MediaItem with metadata from TMDB.
        Uses cache if available and recent, otherwise queries TMDB.
        Updates self fields (new_title, new_year, etc).
        Returns:
            True if enrichment succeeded and metadata found, False if not found or ignored.
        Side effects:
            Updates cache, logs actions.
        """
        if not hasattr(self.config, "_api_calls"):
            self.config._api_calls = 0
        orig_cache_key = self.config.cache_manager.get_cache_key(self)
        cached = self.config.cache.get(orig_cache_key)
        should_skip = (
            not self.config.dry_run and cached and is_recent(cached.get("last_checked", ""), self.config)
        )
        self.skipped_by_cache = should_skip
        found_by_filename = None
        if not cached and not self.tmdb_id and self.files:
            filenames_set = set(os.path.basename(f) for f in self.files)
            for entry in self.config.cache.values():
                cfnames = entry.get("current_filenames")
                ofnames = entry.get("original_filenames")
                if cfnames and isinstance(cfnames, list):
                    if filenames_set & set(cfnames):
                        found_by_filename = entry
                        break
                elif ofnames and isinstance(ofnames, list):
                    if filenames_set & set(ofnames):
                        found_by_filename = entry
                        break
        if not cached and not self.tmdb_id and found_by_filename:
            cached = found_by_filename
            should_skip = (
                not self.config.dry_run and cached and is_recent(cached.get("last_checked", ""), self.config)
            )
            self.skipped_by_cache = should_skip
        if cached and cached.get("status") == "ignored":
            log.debug(f"📦 Skipping (ignored in cache for: {self.original_title} ({self.original_year}))")
            return False
        if (
            cached
            and cached.get("status") == "not_found"
            and is_recent(cached.get("last_checked", ""), self.config)
        ):
            log.debug(
                f"📦 Skipping (recent not_found cache for: {self.original_title} ({self.original_year}))"
            )
            return False
        if not self.config.quiet and not should_skip:
            log.info("")
        if should_skip:
            if cached.get("status") == "not_found":
                log.debug(f"📦 Skipping (previously not found): {self.original_title} ({self.original_year})")
                return False
            if cached.get("status") == "ignored":
                log.debug(f"📦 Skipping (ignored in cache): {self.original_title} ({self.original_year})")
                return False
            self.new_title = cached.get("title")
            self.new_year = cached.get("year")
            self.new_tmdb_id = cached.get("tmdb_id")
            self.new_tvdb_id = cached.get("tvdb_id")
            self.new_imdb_id = cached.get("imdb_id")
            log.debug(
                f"📦 Used cached metadata for {self.new_title or self.original_title} ({self.new_year or self.original_year})"
            )
            return True
        log.debug(
            f"[CACHE DEBUG] Enrich for {self.original_title} ({self.original_year}) | cache_key: {orig_cache_key} | found: {bool(cached)} | recent: {is_recent(cached.get('last_checked', ''), self.config) if cached else 'N/A'} | dry_run: {self.config.dry_run} | should_skip: {should_skip}"
        )
        self.config._api_calls += 1
        result = self.config.tmdb_query_service.query(self, self.type)
        if not result:
            orig_cache_key = self.config.cache_manager.get_cache_key(self)
            cache_key = orig_cache_key
            self._update_and_save_cache(
                cache_key,
                {
                    "title": self.title,
                    "year": self.year,
                    "type": self.type,
                    "last_checked": datetime.now().isoformat(),
                    "no_result": True,
                    "status": "not_found",
                },
            )
            self.match_failed = True
            return False
        if hasattr(result, "id"):
            if result.id and (self.tmdb_id is None or result.id != self.tmdb_id):
                log.warning(f"  ⚠️ TMDB ID mismatch: {self.tmdb_id} → {result.id}")
            self.new_tmdb_id = result.id
        if hasattr(result, "tvdb_id"):
            if getattr(result, "tvdb_id", None) and (self.tvdb_id is None or result.tvdb_id != self.tvdb_id):
                log.warning(f"  ⚠️ TVDB ID mismatch: {self.tvdb_id} → {result.tvdb_id}")
            self.new_tvdb_id = result.tvdb_id
        if hasattr(result, "imdb_id"):
            if getattr(result, "imdb_id", None) and (self.imdb_id is None or result.imdb_id != self.imdb_id):
                log.warning(f"  ⚠️ IMDB ID mismatch: {self.imdb_id} → {result.imdb_id}")
            self.new_imdb_id = result.imdb_id
        tmdb_title = getattr(result, "title", getattr(result, "name", None))
        tmdb_date = getattr(result, "first_air_date", getattr(result, "release_date", None))
        if tmdb_date:
            res_year = tmdb_date.year if hasattr(tmdb_date, "year") else int(str(tmdb_date)[:4])
            if res_year != self.year:
                self.new_year = res_year
        if tmdb_title and tmdb_title != self.title:
            self.new_title = tmdb_title
        self._update_and_save_cache(
            orig_cache_key,
            {
                "last_checked": datetime.now().isoformat(),
                "type": self.type,
                "tmdb_id": self.new_tmdb_id or self.tmdb_id,
                "tvdb_id": self.new_tvdb_id or self.tvdb_id,
                "imdb_id": self.new_imdb_id or self.imdb_id,
                "year": self.new_year or self.year,
                "title": self.new_title or self.title,
                "rename_history": [],
                "status": "found",
            },
        )
        new_cache_key = self.config.cache_manager.get_cache_key(self)
        if new_cache_key != orig_cache_key:
            self.config.cache[new_cache_key] = self.config.cache.pop(orig_cache_key)
            self._update_and_save_cache(new_cache_key, self.config.cache[new_cache_key])
        return True

    def needs_rename(self) -> bool:
        """
        Return True if any new_* fields are set, indicating a rename is needed.
        """
        return any(
            [
                self.new_title,
                self.new_year,
                self.new_tmdb_id,
                self.new_tvdb_id,
                self.new_imdb_id,
            ]
        )

    def filenames(self) -> list[tuple[str, str]]:
        """
        Generate (old_filename, new_filename) tuples for each file.
        Returns:
            List of (old_filename, new_filename) tuples.
        """
        ops: list[tuple[str, str]] = []
        for file_path in self.files:
            _, old = os.path.split(file_path)
            new = generate_new_filename(self, old)
            ops.append((old, new))
        return ops


def strip_tags_and_rename(filenames, dry_run=False, log=None):
    """
    For each file in `filenames`, strip all {tmdb-*}, {tvdb-*}, {imdb-*} tags from the basename.
    If `dry_run` is False, actually rename the files. Returns the new filenames.
    Accepts a single filename (str) or a list of filenames.
    """

    def _strip_all_tags(basename):
        return re.sub(r"\{(?:tmdb|tvdb|imdb)-[^}]+\}", "", basename).strip()

    if isinstance(filenames, str):
        filenames = [filenames]
    updated = []
    for fname in filenames:
        dirpath = os.path.dirname(fname)
        new_basename = _strip_all_tags(os.path.basename(fname))
        new_path = os.path.join(dirpath, new_basename)
        if fname != new_path:
            if os.path.exists(fname) and not dry_run:
                try:
                    os.rename(fname, new_path)
                    log.info(f"Stripped tags: {fname} → {new_path}", "YELLOW")
                except Exception as e:
                    log.error(f"Failed to rename {fname}: {e}", "RED")
            else:
                log.info(f"Would strip tags: {fname} → {new_path} (dry run)", "YELLOW")
            updated.append(new_path)
        else:
            updated.append(fname)
    return updated if len(updated) > 1 else updated[0]

def normalize_with_aliases(string: str) -> str:
    """
    Normalize a string to ASCII, lowercased, with canonical abbreviation/alias mapping.
    """

    CANONICAL_ALIASES = {
        "&": "and",
        "and": "and",
        "vs.": "versus",
        "vs": "versus",
        "ep.": "episode",
        "ep": "episode",
        "vol.": "volume",
        "vol": "volume",
        "pt.": "part",
        "pt": "part",
        "dr.": "doctor",
        "dr": "doctor",
        "doctor": "doctor",
    }

    def remove_apostrophes(s: str) -> str:
        return re.sub(r"[’'`ʹʼ]", "", s)

    def to_ascii(s: str) -> str:
        nfkd = unicodedata.normalize("NFKD", s)
        return nfkd.encode("ASCII", "ignore").decode()

    def canonicalize_tokens(s: str) -> str:
        # Tokenize, but preserve non-word separators
        words = re.split(r"(\W+)", s)
        normalized = [
            CANONICAL_ALIASES.get(w.lower(), w.lower()) if w.strip() else w
            for w in words
        ]
        return "".join(normalized)

    def to_lower_strip_spaces(s: str) -> str:
        return re.sub(r"\s+", " ", s).strip().lower()

    s = string
    s = remove_apostrophes(s)
    s = to_ascii(s)
    s = canonicalize_tokens(s)
    s = to_lower_strip_spaces(s)
    return s


def parse_file_group(config: "IdarrConfig", base_name: str, files: list[str]) -> MediaItem:
    """
    Parse a group of files and return a MediaItem instance.
    Uses cache entry by filename to fill missing IDs if available.
    Args:
        config: IdarrConfig.
        base_name: Base name for the group.
        files: List of filenames in the group.
    Returns:
        MediaItem instance.
    """

    title = re.sub(r"{(tmdb|tvdb|imdb)-[^}]+}", "", base_name)
    title = re.sub(YEAR_REGEX, "", title).strip()
    year_match = YEAR_REGEX.search(base_name)
    year = int(year_match.group(1)) if year_match else None
    tmdb_match = TMDB_ID_REGEX.search(base_name)
    tmdb_id = int(tmdb_match.group(1)) if tmdb_match else None
    tvdb_match = TVDB_ID_REGEX.search(base_name)
    tvdb_id = int(tvdb_match.group(1)) if tvdb_match else None
    imdb_match = IMDB_ID_REGEX.search(base_name)
    imdb_id = imdb_match.group(1) if imdb_match else None
    file_paths = sorted([os.path.join(config.source_dir, file) for file in files if not file.startswith(".")])
    is_series = any(SEASON_PATTERN.search(f) for f in files) or (tvdb_id is not None)
    is_collection = (year is None) and not is_series

    id_fields = {"tmdb_id": tmdb_id, "tvdb_id": tvdb_id, "imdb_id": imdb_id}
    if any(v is None for v in id_fields.values()):

        filename_set = set(os.path.basename(f) for f in files)
        found_cache = None
        for entry in config.cache.values():
            cfnames = entry.get("current_filenames")
            ofnames = entry.get("original_filenames")
            if cfnames and isinstance(cfnames, list):
                if filename_set & set(cfnames):
                    found_cache = entry
                    break
            elif ofnames and isinstance(ofnames, list):
                if filename_set & set(ofnames):
                    found_cache = entry
                    break
        if found_cache:

            tmdb_id = tmdb_id or found_cache.get("tmdb_id")
            tvdb_id = tvdb_id or found_cache.get("tvdb_id")
            imdb_id = imdb_id or found_cache.get("imdb_id")

    cache_type = None
    tmp_media_item = None
    tmp_data = {
        "type": "movie",
        "title": title,
        "year": year,
        "tmdb_id": tmdb_id,
        "tvdb_id": tvdb_id,
        "imdb_id": imdb_id,
        "files": file_paths,
    }
    cache_entry = None
    if tmdb_id is not None:
        for possible_type in ("movie", "tv_series", "collection"):
            tmp_data["type"] = possible_type
            tmp_media_item = MediaItem(**tmp_data, config=config)
            possible_cache = config.cache.get(config.cache_manager.get_cache_key(tmp_media_item))
            if possible_cache:
                cache_entry = possible_cache
                cache_type = possible_cache.get("type")
                break
    else:
        tmp_media_item = MediaItem(**tmp_data, config=config)
        cache_entry = config.cache.get(config.cache_manager.get_cache_key(tmp_media_item))
        if cache_entry:
            cache_type = cache_entry.get("type")

    if cache_type == "collection" or is_collection:
        data = {
            "type": "collection",
            "title": title,
            "year": None,
            "imdb_id": imdb_id,
            "tmdb_id": tmdb_id,
            "files": file_paths,
        }
    elif cache_type == "tv_series" or is_series or tvdb_id:
        data = {
            "type": "tv_series",
            "title": title,
            "year": year,
            "tvdb_id": tvdb_id,
            "imdb_id": imdb_id,
            "tmdb_id": tmdb_id,
            "files": file_paths,
        }
    else:
        data = {
            "type": "movie",
            "title": title,
            "year": year,
            "tmdb_id": tmdb_id,
            "imdb_id": imdb_id,
            "files": file_paths,
        }

    media_item = MediaItem(**data, config=config)
    if cache_entry:
        for field in ("tmdb_id", "tvdb_id", "imdb_id"):
            val = cache_entry.get(field)
            if val:
                setattr(media_item, field, val)
                new_field = f"new_{field}"
                if not getattr(media_item, new_field, None):
                    setattr(media_item, new_field, val)
    return media_item


def scan_files_in_flat_folder(config: "IdarrConfig") -> list[MediaItem]:
    """
    Scan a flat folder for image assets and group them into MediaItem instances.
    Args:
        config: IdarrConfig with source_dir.
    Returns:
        List of MediaItem instances representing asset groups.
    Side effects:
        May delete non-image files if configured.
        Logs progress.
    """
    log.info(f"📂 Scanning directory for image assets: {config.source_dir}")
    try:
        files = os.listdir(config.source_dir)
    except FileNotFoundError:
        return []
    groups = defaultdict(list)
    assets_dict = []

    already_renamed = {
        entry.get("current_filename") for entry in config.cache.values() if entry.get("current_filename")
    }
    for file in files:
        if file.startswith("."):
            continue
        ext = os.path.splitext(file)[-1].lower()
        if ext not in IMAGE_EXTENSIONS and config.remove_non_image_files:
            full_path = os.path.join(config.source_dir, file)
            if config.source_dir:
                if config.dry_run:
                    log.info(f"[DRY RUN] Would delete non-image file: {file}")
                else:
                    try:
                        os.remove(full_path)
                        log.info(f"🗑️ Removed non-image file: {file}")
                    except Exception as e:
                        log.error(f"❌ Failed to delete {file}: {e}")
            continue

        if os.path.basename(file) in already_renamed:
            continue
        title = file.rsplit(".", 1)[0]
        raw_title = SEASON_PATTERN.split(title)[0].strip()
        groups[raw_title].append(file)
    groups = dict(sorted(groups.items(), key=lambda x: x[0].lower()))

    all_files = [file for group in groups.values() for file in group if not file.startswith(".")]
    global progress_context, progress_bar
    progress_bar = Progress(
        TextColumn("{task.description}"),
        BarColumn(bar_width=None),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        expand=True,
        console=console,
    )
    progress_context = progress_bar.__enter__()
    try:
        task = progress_bar.add_task(
            f"Processing files {os.path.basename(config.source_dir)}...", total=len(all_files)
        )
        for base_name, files in groups.items():
            assets_dict.append(parse_file_group(config, base_name, files))
            progress_bar.update(task, advance=len(files))
    finally:
        flush_status()
    total_assets = sum(len(v) for v in groups.values())
    log.info(
        f"✅ Completed scanning: discovered {len(assets_dict)} asset groups covering {total_assets} files"
    )
    return assets_dict


def handle_data(config: "IdarrConfig", items: list["MediaItem"]) -> list["MediaItem"]:
    """
    Enrich a list of MediaItem objects with metadata from TMDB.
    Updates UNMATCHED_CASES and TVDB_MISSING_CASES as needed.
    Args:
        config: IdarrConfig.
        items: List of MediaItem objects to enrich.
    Returns:
        List of MediaItem objects (with fields populated).
    Side effects:
        Updates cache, logs, updates pending matches.
    """
    ignored_count = 0
    total_items = len(items)

    if total_items > 0:
        log.info("🔄 Starting metadata enrichment via TMDB")

    pending_matches = getattr(config, "pending_matches", {}).copy()
    ignored_title_keys = set(getattr(config, "ignored_title_keys", set()))
    keys_to_upsert = []

    with console.status(f"[green]Processing 1/{total_items} items...") as status:
        for idx, item in enumerate(items, 1):
            raw_title_key = f"{item.title} ({item.year})" if item.year else item.title
            cache_key = config.cache_manager.get_cache_key(item)

            if is_ignored(item, ignored_title_keys):
                log.debug(f"⏭️ Ignored by user-defined exclusions: {raw_title_key}")
                if raw_title_key in pending_matches:
                    del pending_matches[raw_title_key]
                ignored_count += 1
                if idx % 10 == 0 or idx == total_items:
                    status.update(f"[green]Processing {idx}/{total_items} items...")
                continue

            if getattr(config, "skip_collections", False) and getattr(item, "type", None) == "collection":
                log.info(f"⏭️  Skipped collection: {item.title}")
                continue

            enriched = item.enrich()
            cache_key = config.cache_manager.get_cache_key(item)

            if not enriched:
                item.match_failed = True
                UNMATCHED_CASES.append(
                    {
                        "media_type": item.type,
                        "title": item.title,
                        "year": item.year if item.year is not None else "",
                        "tmdb_id": getattr(item, "tmdb_id", ""),
                        "tvdb_id": getattr(item, "tvdb_id", ""),
                        "imdb_id": getattr(item, "imdb_id", ""),
                        "files": ";".join(item.files),
                        "match_reason": getattr(item, "match_reason", ""),
                    }
                )
                log.debug(f"Upsert NOT_FOUND: {item.title} ({item.year})")
                config.cache_manager.upsert(cache_key, {"status": "not_found"}, item)
                keys_to_upsert.append(cache_key)
                if raw_title_key not in pending_matches:
                    pending_matches[raw_title_key] = "add_tmdb_url_here"
                if idx % 10 == 0 or idx == total_items:
                    status.update(f"[green]Processing {idx}/{total_items} items...")
                continue
            else:
                new_key = config.cache_manager.get_cache_key(item)
                found_cache = config.cache.get(new_key)
                meta_changed = (
                    found_cache.get("title") != (item.new_title or item.title)
                    or found_cache.get("year") != (item.new_year if item.new_year is not None else item.year)
                    or found_cache.get("tmdb_id") != (item.new_tmdb_id or item.tmdb_id)
                    or found_cache.get("tvdb_id") != (item.new_tvdb_id or item.tvdb_id)
                    or found_cache.get("imdb_id") != (item.new_imdb_id or item.imdb_id)
                )
                is_missing = found_cache is None
                is_stale = (
                    found_cache is not None
                    and not config.dry_run
                    and not is_recent(found_cache.get("last_checked", ""), config)
                )
                if is_missing or is_stale or meta_changed:
                    log.debug(
                        f"Upsert FOUND: {item.title} ({item.year}), "
                        f"is_missing={is_missing}, is_stale={is_stale}, meta_changed={meta_changed}"
                    )
                    config.cache_manager.upsert(new_key, {"status": "found"}, item)
                    keys_to_upsert.append(new_key)
                else:
                    log.debug(f"NO UPSERT NEEDED: {item.title} ({item.year}) - fully cached and unchanged.")

            if idx % 1 == 0 or idx == total_items:
                status.update(f"[green]Processing {idx}/{total_items} items...")

            if enriched and raw_title_key in pending_matches:
                del pending_matches[raw_title_key]
            if item.type == "tv_series":
                has_tvdb = getattr(item, "tvdb_id", None) or getattr(item, "new_tvdb_id", None)
                if not has_tvdb:
                    TVDB_MISSING_CASES.append(
                        {
                            "title": item.title,
                            "year": item.year,
                            "tmdb_id": getattr(item, "tmdb_id", ""),
                            "imdb_id": getattr(item, "imdb_id", ""),
                            "files": ";".join(item.files),
                        }
                    )

    new_pending = update_pending_matches_from_cache(config)
    save_pending_matches(new_pending)
    config.pending_matches = new_pending

    if config.show_unmatched and UNMATCHED_CASES:
        for case in UNMATCHED_CASES:
            title = case.get("title", "Unknown")
            log.warning(f"❌ Unmatched: {title}")
    if total_items > 0 or ignored_count > 0:
        log.info(f"✅ Completed metadata enrichment ({ignored_count} item(s) ignored by exclusion list)")
    return items


def generate_new_filename(media_item: "MediaItem", old_filename: str) -> str:
    """
    Generate a new filename for a media item based on its (possibly updated) metadata.
    Args:
        media_item: MediaItem with new_* fields.
        old_filename: Original filename (with extension).
    Returns:
        New filename string (with extension), cleaned for filesystem.
    """
    old_name_no_ext = os.path.splitext(old_filename)[0]
    base_title = (
        media_item.new_title
        if media_item.new_title
        else (media_item.title if media_item.title else old_name_no_ext)
    )
    base_year = media_item.new_year if media_item.new_year is not None else media_item.year

    id_parts = []
    for attr, prefix in (("tmdb_id", "tmdb"), ("tvdb_id", "tvdb"), ("imdb_id", "imdb")):
        val = getattr(media_item, f"new_{attr}", None)
        if val is None:
            val = getattr(media_item, attr, None)

        if prefix == "imdb" and (not isinstance(val, str) or not val.startswith("tt")):
            continue
        if val:
            id_parts.append(f"{prefix}-{val}")
    suffix = "".join(f" {{{part}}}" for part in id_parts)

    season_suffix = ""
    match = SEASON_PATTERN.search(old_filename)
    if match:
        season_suffix = match.group(0)

    name, ext = os.path.splitext(old_filename)
    base = f"{base_title}{f' ({base_year})' if base_year else ''}"

    if media_item.type == "tv_series":
        new_name = f"{base}{suffix}{season_suffix}{ext}"
    else:
        new_name = f"{base}{season_suffix}{suffix}{ext}"

    forbidden = r'[<>:"/\\|?*\x00-\x1F]'
    cleaned = re.sub(forbidden, "", new_name)
    cleaned = " ".join(cleaned.split()).strip()
    return cleaned


def is_ignored(media_item, ignored_title_keys):
    """
    Returns True if the media_item should be ignored (by raw title key), False otherwise.
    """
    raw_title_key = f"{media_item.title} ({media_item.year})" if media_item.year else media_item.title
    return raw_title_key in ignored_title_keys


def rename_files(
    items: list["MediaItem"], config: "IdarrConfig"
) -> tuple[list[tuple[str, str, str]], list[dict[str, Any]]]:
    """
    Rename files for all enriched MediaItem objects, respecting DRY_RUN mode.
    Handles filename conflicts, length limits, and logs all actions.
    In case of a conflict, keeps the file with the newest creation date,
    moves the older file to a global `duplicates` folder next to idarr.py, and logs this in a CSV.
    Returns grouped diff-style output once per media item.
    """
    mode = "DRY RUN" if config.dry_run else "LIVE"
    non_skipped_items = [item for item in items if not getattr(item, "match_failed", False)]
    if len(non_skipped_items) > 0:
        log.info(f"🏷  Starting file rename process ({mode} mode)")
    file_updates: list[tuple[str, str, str]] = []
    duplicate_log: list[dict[str, Any]] = []
    renamed, skipped = 0, 0

    duplicates_dir = os.path.join(SCRIPT_DIR, "duplicates")
    os.makedirs(duplicates_dir, exist_ok=True)

    with console.status("[cyan]Renaming files... Please wait.", spinner="dots"):
        for media_item in items:
            # User-defined ignore check
            if is_ignored(media_item, getattr(config, "ignored_title_keys", set())):
                log.debug(
                    f"⏭️ Ignored by user-defined exclusions (rename skipped): {media_item.title} ({media_item.year})"
                    if media_item.year
                    else media_item.title
                )
                continue

            if getattr(media_item, "match_failed", False):
                continue

            mtype = media_item.type
            title = media_item.new_title or media_item.title
            year = media_item.new_year if media_item.new_year is not None else media_item.year

            if mtype == "tv_series":
                label = "Series"
            elif mtype == "collection":
                label = "Collection"
            else:
                label = "Movie"

            prefix = "[DRY RUN] " if config.dry_run else ""
            header = f"{prefix}{label}: {title}"
            if year:
                header += f" ({year})"

            key = config.cache_manager.get_cache_key(media_item)

            item_renames = []

            for file_path in media_item.files:
                directory, old_filename = os.path.split(file_path)
                new_filename = generate_new_filename(media_item, old_filename)
                new_path = os.path.join(directory, new_filename)

                if len(new_filename) > 255:
                    log.warning(f"⛔ Skipped (too long): {new_filename}")
                    skipped += 1
                    continue

                if os.path.exists(new_path) and old_filename.lower() != new_filename.lower():
                    try:
                        src_stat = os.stat(file_path)
                    except FileNotFoundError:
                        log.warning(f"❌ Source file not found, skipping: {file_path}")
                        skipped += 1
                        continue
                    dst_stat = os.stat(new_path)
                    src_ctime = getattr(src_stat, "st_ctime", src_stat.st_mtime)
                    dst_ctime = getattr(dst_stat, "st_ctime", dst_stat.st_mtime)

                    if src_ctime >= dst_ctime:
                        move_src = new_path
                        move_dst_name = new_filename
                        keep_path = file_path
                    else:
                        move_src = file_path
                        move_dst_name = old_filename
                        keep_path = new_path

                    if not os.path.exists(keep_path):
                        log.warning(
                            f"❗ [SAFEGUARD] Conflict detected but 'original' file '{os.path.basename(keep_path)}' does not exist in source. "
                            f"Skipping move of '{os.path.basename(move_src)}' to duplicates to prevent data loss."
                        )
                        skipped += 1
                        continue

                    dest = os.path.join(duplicates_dir, os.path.basename(move_dst_name))
                    if os.path.exists(dest):
                        base, ext = os.path.splitext(os.path.basename(move_dst_name))
                        suffix = int(time.time())
                        dest = os.path.join(duplicates_dir, f"{base}_{suffix}{ext}")

                    if not config.dry_run:
                        try:
                            shutil.move(move_src, dest)
                            log.warning(f"🗂️ Duplicate moved: {move_dst_name} → {dest}")
                            duplicate_log.append(
                                {
                                    "action": "moved",
                                    "kept_file": os.path.basename(keep_path),
                                    "kept_path": keep_path,
                                    "kept_ctime": src_ctime if move_src == new_path else dst_ctime,
                                    "moved_file": os.path.basename(move_dst_name),
                                    "moved_path": dest,
                                    "moved_ctime": src_ctime if move_src == file_path else dst_ctime,
                                }
                            )
                        except Exception as e:
                            log.error(f"❌ Failed to move duplicate '{move_dst_name}': {e}")
                            skipped += 1
                            continue
                    else:
                        log.info(f"[DRY RUN] Would move duplicate: {move_dst_name} → {dest}", "YELLOW")

                    if src_ctime < dst_ctime:
                        continue

                if old_filename != new_filename:
                    item_renames.append((old_filename, new_filename))
                    file_updates.append((media_item.type, old_filename, new_filename))
                    renamed += 1

                    if not config.dry_run:
                        try:
                            os.rename(file_path, new_path)
                            cache_key = key
                            cache_entry = config.cache.get(cache_key, {})
                            hist = cache_entry.get("rename_history", [])
                            hist.append(
                                {
                                    "from": old_filename,
                                    "to": new_filename,
                                    "timestamp": datetime.now().isoformat(),
                                }
                            )
                            cache_entry["rename_history"] = hist

                            origs = set(cache_entry.get("original_filenames", []))
                            origs.add(old_filename)
                            cache_entry["original_filenames"] = sorted(origs)

                            real_files = (
                                set(os.listdir(config.source_dir))
                                if os.path.isdir(config.source_dir)
                                else set()
                            )
                            possibles = set(cache_entry["original_filenames"]) | {h["to"] for h in hist}
                            cache_entry["current_filenames"] = sorted(f for f in possibles if f in real_files)

                            config.cache[cache_key] = cache_entry
                        except Exception as e:
                            log.error(f"file_path: {file_path}")
                            log.error(f"new_path: {new_path}")
                            log.error(f"❌ Failed to rename {old_filename}: {e}")
                            skipped += 1

            if item_renames:
                log.info(header)
                for old_filename, new_filename in item_renames:
                    log.info(f"        - {old_filename}", "RED")
                    log.info(f"        + {new_filename}", "GREEN")

    return file_updates, duplicate_log


def flush_status():
    global status_context, progress_context, progress_bar
    if status_context is not None:
        try:
            status_context.__exit__(None, None, None)
        except Exception:
            pass
        status_context = None
    if progress_context is not None:
        try:
            progress_context.__exit__(None, None, None)
        except Exception:
            pass
        progress_context = None
        progress_bar = None


def prune_orphaned_cache_entries(config: IdarrConfig) -> None:
    global status_context
    log.info("🧹 Starting prune operation for orphaned cache entries...")

    try:
        current_files = set(os.listdir(config.source_dir))
    except Exception as e:
        log.error(f"Failed to list source directory: {e}")
        return

    # Load pending matches
    pending_file = PENDING_MATCHES_PATH
    pending_matches = {}
    if os.path.exists(pending_file):
        with open(pending_file, encoding="utf-8") as f:
            lines = [line for line in f if not line.lstrip().startswith("//")]
            try:
                pending_matches = json.loads("".join(lines))
            except Exception as e:
                log.warning(f"⚠️ Failed to load pending matches: {e}")
                pending_matches = {}

    removed_keys = []
    removed_titles = []
    cache_items = list(config.cache.items())
    total = len(cache_items)
    entries_update = 10

    status_context = console.status("[cyan]Pruning orphaned cache entries...", spinner="dots")
    status = status_context.__enter__()
    for idx, (key, entry) in enumerate(cache_items, 1):
        originals = set(entry.get("original_filenames", []))
        currents = set(entry.get("current_filenames", []))
        relevant = originals | currents
        if not (relevant & current_files):
            with sqlite3.connect(config.cache_manager.db_path) as conn:
                conn.execute("DELETE FROM cache WHERE key = ?", (key,))
                conn.commit()
            removed_keys.append(key)
            title = entry.get("title")
            year = entry.get("year")
            if year:
                removed_titles.append(f"{title} ({year})")
            else:
                removed_titles.append(title)
        if idx % entries_update == 0 or idx == total:
            status.update(f"[cyan]Searching, Please wait... ({idx:,}/{total:,})")

    for key in removed_keys:
        config.cache.pop(key, None)
        config.cache_manager.cache.pop(key, None)

    # Remove pruned titles from pending_matches
    pending_removed = 0
    for removed_title in removed_titles:
        if removed_title in pending_matches:
            del pending_matches[removed_title]
            log.info(f"🗑️ Pruned orphaned pending match: {removed_title}")
            pending_removed += 1

    # Save updated pending matches
    save_pending_matches(pending_matches, pending_file)

    log.info(
        f"✅ Prune operation complete. {len(removed_keys)} entries removed. {pending_removed} pending matches cleaned up."
    )
    flush_status()


def print_rich_help() -> None:
    table = Table(show_header=True, header_style="bold purple")
    table.add_column("Option", style="green", no_wrap=True)
    table.add_column("Description", style="white")
    table.add_column("Default Value", style="yellow")

    # --- General Options ---
    table.add_row("[bold]General Options[/bold]", "", "")
    table.add_row("--source DIR", "Directory of input image files", "Required/.env file")
    table.add_row("--tmdb-api-key KEY", "TMDB API key override", "Required/.env file")
    table.add_row("--dry-run", "Simulate changes (no actual file ops)", "False")
    table.add_row("--quiet", "Suppress output except progress bars", "False")
    table.add_row("--debug", "Enable debug logging", "False")
    table.add_row("--limit N", "Maximum items to process", "0 (unlimited)")
    table.add_row("--remove-non-image-files", "Remove non-image files", "False")
    table.add_row("--pending-matches", "Only process pending matches list", "False")
    table.add_row("--ignore-file PATH", "Path to ignored_titles.jsonc", "logs/ignored_titles.jsonc")
    table.add_row(
        "--pending-matches-path PATH", "Path to pending_matches.jsonc", "logs/pending_matches.jsonc"
    )
    table.add_section()

    # --- Caching Options ---
    table.add_row("[bold]Caching Options[/bold]", "", "")
    table.add_row("--frequency-days DAYS", "Days before cache considered stale", "30")
    table.add_row("--tvdb-frequency DAYS", "Days before retry for missing TVDb IDs", "7")
    table.add_row("--clear-cache", "Delete cache before running", "False")
    table.add_row("--no-cache", "Skip loading/saving cache", "False")
    table.add_row("--prune", "Prune orphaned cache entries", "False")
    table.add_row("--purge", 'Delete cache by TMDB ID, "Title (Year)", or "Title"', "")
    table.add_row("--cache-path PATH", "Custom cache file path", "cache/idarr_cache.db")
    table.add_section()

    # --- Filtering Options ---
    table.add_row("[bold]Filtering Options[/bold]", "", "")
    table.add_row("--filter", "Enable filtering mode", "")
    table.add_row("--type {movie,tv_series,collection}", "Only process a specific media type", "")
    table.add_row("--year YEAR", "Only process items from this year", "")
    table.add_row("--contains TEXT", "Only include titles containing text", "")
    table.add_row("--id ID", "Only items with a specific TMDB/TVDB/IMDB ID", "")
    table.add_row("--skip-collections", "Skip collection enrichment", "False")
    table.add_section()

    # --- Export & Recovery ---
    table.add_row("[bold]Export & Recovery[/bold]", "", "")
    table.add_row("--show-unmatched", "Show unmatched items", "False")
    table.add_row("--revert", "Undo file renames using cache", "False")
    table.add_section()

    # --- Other ---
    table.add_row("[bold]Other[/bold]", "", "")
    table.add_row("-h, --help", "Show this help message and exit", "")
    table.add_row("--version", "Show program's version and exit", "")

    console.print("[bold cyan]IDARR: Poster Asset Renamer & ID Tagger[/bold cyan]")
    console.print(table)
    console.print("[bold]Examples:[/bold]")
    console.print("  python idarr.py --source ./images --dry-run")
    console.print("  python idarr.py --filter --type movie --year 2023")


def env_bool(name: str, default: bool = False) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.lower() in ("1", "true", "yes", "on")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enrich and rename media image files using TMDB metadata.",
        add_help=False,
    )
    parser.add_argument("--version", action="version", version=f"idarr.py {FULL_VERSION}")
    parser.add_argument("-h", "--help", action="store_true", help=argparse.SUPPRESS)

    # --- General Options ---
    general = parser.add_argument_group("General Options")
    general.add_argument(
        "--source",
        metavar="DIR",
        type=str,
        default=os.environ.get("SOURCE_DIR"),
        help=argparse.SUPPRESS,
    )
    general.add_argument(
        "--tmdb-api-key",
        metavar="KEY",
        type=str,
        default=os.environ.get("TMDB_API_KEY"),
        help=argparse.SUPPRESS,
    )
    general.add_argument(
        "--dry-run",
        action="store_true",
        default=env_bool("DRY_RUN", False),
        help=argparse.SUPPRESS,
    )
    general.add_argument(
        "--quiet",
        action="store_true",
        default=env_bool("QUIET", False),
        help=argparse.SUPPRESS,
    )
    general.add_argument(
        "--debug",
        action="store_true",
        default=env_bool("DEBUG", False),
        help=argparse.SUPPRESS,
    )
    general.add_argument(
        "--limit",
        metavar="N",
        type=int,
        default=int(os.environ.get("LIMIT")) if os.environ.get("LIMIT") else None,
        help=argparse.SUPPRESS,
    )
    general.add_argument(
        "--remove-non-image-files",
        action="store_true",
        default=env_bool("REMOVE_NON_IMAGE_FILES", False),
        help=argparse.SUPPRESS,
    )
    general.add_argument(
        "--ignore-file",
        metavar="PATH",
        type=str,
        default=os.environ.get("IGNORE_FILE", os.path.join(LOG_DIR, "ignored_titles.jsonc")),
        help=argparse.SUPPRESS,
    )
    general.add_argument(
        "--pending-matches",
        action="store_true",
        default=env_bool("PENDING_MATCHES", False),
        help="Only process and resolve pending matches (including renaming just-resolved entries).",
    )
    general.add_argument(
        "--pending-matches-path",
        metavar="PATH",
        type=str,
        default=os.environ.get("PENDING_MATCHES_PATH", PENDING_MATCHES_PATH),
        help=argparse.SUPPRESS,
    )

    # --- Caching Options ---
    cache = parser.add_argument_group("Caching Options")
    cache.add_argument(
        "--frequency-days",
        metavar="DAYS",
        type=int,
        default=int(os.environ.get("FREQUENCY_DAYS", "30")),
        help=argparse.SUPPRESS,
    )
    cache.add_argument(
        "--tvdb-frequency",
        metavar="DAYS",
        type=int,
        default=int(os.environ.get("TVDB_FREQUENCY", "7")),
        help=argparse.SUPPRESS,
    )
    cache.add_argument(
        "--clear-cache",
        action="store_true",
        default=env_bool("CLEAR_CACHE", False),
        help=argparse.SUPPRESS,
    )
    cache.add_argument(
        "--cache-path",
        metavar="PATH",
        type=str,
        default=os.environ.get("CACHE_PATH", CACHE_PATH),
        help=argparse.SUPPRESS,
    )
    cache.add_argument(
        "--no-cache",
        action="store_true",
        default=env_bool("NO_CACHE", False),
        help=argparse.SUPPRESS,
    )
    cache.add_argument(
        "--prune",
        action="store_true",
        default=env_bool("PRUNE", False),
        help="Prune cache entries not associated with any file in the source directory.",
    )
    cache.add_argument(
        "--purge",
        type=str,
        default=os.environ.get("PURGE"),
        help='Delete cache entries by TMDB ID or by "Title (Year)" or "Title"',
    )

    # --- Filtering Options ---
    filtering = parser.add_argument_group("Filtering Options")
    filtering.add_argument(
        "--filter",
        action="store_true",
        default=env_bool("FILTER", False),
        help=argparse.SUPPRESS,
    )
    filtering.add_argument(
        "--type",
        choices=["movie", "tv_series", "collection"],
        default=os.environ.get("TYPE"),
        help=argparse.SUPPRESS,
    )
    filtering.add_argument(
        "--year",
        metavar="YEAR",
        type=int,
        default=int(os.environ.get("YEAR")) if os.environ.get("YEAR") else None,
        help=argparse.SUPPRESS,
    )
    filtering.add_argument(
        "--contains",
        metavar="TEXT",
        type=str,
        default=os.environ.get("CONTAINS"),
        help=argparse.SUPPRESS,
    )
    filtering.add_argument(
        "--id",
        metavar="ID",
        type=str,
        default=os.environ.get("ID"),
        help=argparse.SUPPRESS,
    )
    filtering.add_argument(
        "--skip-collections",
        action="store_true",
        default=env_bool("SKIP_COLLECTIONS", False),
        help=argparse.SUPPRESS,
    )

    # --- Export & Recovery ---
    extra = parser.add_argument_group("Export & Recovery")
    extra.add_argument(
        "--show-unmatched",
        action="store_true",
        default=env_bool("SHOW_UNMATCHED", False),
        help=argparse.SUPPRESS,
    )
    extra.add_argument(
        "--revert",
        action="store_true",
        default=env_bool("REVERT", False),
        help=argparse.SUPPRESS,
    )

    args = parser.parse_args()
    if getattr(args, "help", False):
        print_rich_help()
        sys.exit(0)
    return args


def load_runtime_config(args: argparse.Namespace) -> IdarrConfig:
    config = IdarrConfig()

    # General and main runtime flags
    config.dry_run = getattr(args, "dry_run", False)
    config.quiet = getattr(args, "quiet", False)
    config.log_level = "DEBUG" if getattr(args, "debug", False) else "INFO"
    log.configure(quiet=config.quiet, level=config.log_level)

    config.source_dir = getattr(args, "source", None) or os.environ.get("SOURCE_DIR") or ""
    config.tmdb_api_key = getattr(args, "tmdb_api_key", None) or os.environ.get("TMDB_API_KEY") or ""
    config.limit = (
        getattr(args, "limit", None)
        if getattr(args, "limit", None) is not None
        else (int(os.environ.get("LIMIT")) if os.environ.get("LIMIT") else None)
    )
    config.remove_non_image_files = getattr(args, "remove_non_image_files", False)
    config.ignore_file = (
        getattr(args, "ignore_file", None)
        or os.environ.get("IGNORE_FILE")
        or os.path.join(LOG_DIR, "ignored_titles.jsonc")
    )
    config.pending_matches_path = (
        getattr(args, "pending_matches_path", None)
        or os.environ.get("PENDING_MATCHES_PATH")
        or PENDING_MATCHES_PATH
    )

    # Caching
    config.cache_path = (
        getattr(args, "cache_path", None)
        or os.environ.get("CACHE_PATH")
        or os.path.join(SCRIPT_DIR, "cache", "idarr_cache.json")
    )
    config.no_cache = getattr(args, "no_cache", False)
    config.clear_cache = getattr(args, "clear_cache", False)
    config.frequency_days = (
        getattr(args, "frequency_days", None)
        if getattr(args, "frequency_days", None) is not None
        else int(os.environ.get("FREQUENCY_DAYS", "30"))
    )
    config.tvdb_frequency = (
        getattr(args, "tvdb_frequency", None)
        if getattr(args, "tvdb_frequency", None) is not None
        else int(os.environ.get("TVDB_FREQUENCY", "7"))
    )
    config.prune = getattr(args, "prune", False)
    config.purge = getattr(args, "purge", False)

    # Export/Recovery
    config.show_unmatched = getattr(args, "show_unmatched", False)
    config.revert = getattr(args, "revert", False)

    # Filtering (not included in .env, but support CLI as normal)
    config.filter = getattr(args, "filter", False)
    config.type = getattr(args, "type", None)
    config.year = getattr(args, "year", None)
    config.contains = getattr(args, "contains", None)
    config.id = getattr(args, "id", None)
    config.skip_collections = getattr(args, "skip_collections", False)

    # Cache manager
    config.cache_manager = SQLiteCacheManager(
        path=config.cache_path, source_dir=config.source_dir, no_cache=config.no_cache
    )
    if config.clear_cache and os.path.exists(config.cache_path):
        os.remove(config.cache_path)
    config.cache = config.cache_manager.load()

    # TMDB API key prompt (last chance)
    if not config.tmdb_api_key:
        try:
            api_key = Prompt.ask(
                "[bold yellow]TMDB API key not found. Please enter your TMDB API key[/bold yellow]"
            )
            if api_key:
                config.tmdb_api_key = api_key
                save = Prompt.ask("Save this API key to .env for future runs? (y/n)", default="y")
                if save.lower().startswith("y"):
                    dotenv_path = os.path.join(SCRIPT_DIR, ".env")
                    with open(dotenv_path, "a") as f:
                        f.write(f"\nTMDB_API_KEY={api_key}\n")
                    console.print("[green]✅ Saved API key to .env[/green]")
        except Exception as e:
            print("Error prompting for TMDB API key:", e)
    if not config.tmdb_api_key:
        raise RuntimeError(
            "TMDB API key is required. Set via --tmdb-api-key or TMDB_API_KEY environment variable."
        )

    tmdb_client = TMDbAPIs(config.tmdb_api_key)
    config.tmdb_query_service = TMDBQueryService(tmdb_client, config)

    def _flush():
        try:
            flush_status()
            config.cache_manager.save(set(config.cache_manager.cache.keys()))
        except Exception:
            pass

    atexit.register(_flush)

    def _interrupt(signum, frame):
        log.info("\n⏹️  Interrupted by user (Ctrl+C). Exiting gracefully.", "YELLOW")
        _flush()
        sys.exit(130)

    signal.signal(signal.SIGINT, _interrupt)
    signal.signal(signal.SIGTERM, _interrupt)
    return config


def print_settings(config: "IdarrConfig") -> None:
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Section", style="cyan", no_wrap=True)
    table.add_column("Setting", style="green")
    table.add_column("Value", style="white")

    settings = []
    # General
    settings.append(("General", "SOURCE_DIR", str(config.source_dir)))
    settings.append(("General", "DRY_RUN", str(getattr(config, "dry_run", False))))
    settings.append(("General", "QUIET", str(getattr(config, "quiet", False))))
    settings.append(("General", "LOG_LEVEL", str(getattr(config, "log_level", "INFO"))))
    settings.append(("General", "LIMIT", str(getattr(config, "limit", None))))
    settings.append(
        ("General", "REMOVE_NON_IMAGE_FILES", str(getattr(config, "remove_non_image_files", False)))
    )
    settings.append(("General", "IGNORE_FILE", str(getattr(config, "ignore_file", ""))))
    settings.append(("General", "PENDING_MATCHES_PATH", str(PENDING_MATCHES_PATH)))

    # Caching
    settings.append(("Caching", "CACHE_PATH", str(getattr(config, "cache_path", ""))))
    settings.append(("Caching", "NO_CACHE", str(getattr(config, "no_cache", False))))
    settings.append(("Caching", "CLEAR_CACHE", str(getattr(config, "clear_cache", False))))
    settings.append(("Caching", "FREQUENCY_DAYS", str(getattr(config, "frequency_days", 30))))
    settings.append(("Caching", "TVDB_FREQUENCY", str(getattr(config, "tvdb_frequency", 7))))
    settings.append(("Caching", "PRUNE", str(getattr(config, "prune", False))))
    settings.append(("Caching", "PURGE", str(getattr(config, "purge", False))))
    settings.append(("Caching", "CACHE_ENTRIES", str(len(getattr(config, "cache", {})))))

    # Export/Recovery
    settings.append(("Export/Recovery", "SHOW_UNMATCHED", str(getattr(config, "show_unmatched", False))))
    settings.append(("Export/Recovery", "REVERT", str(getattr(config, "revert", False))))

    # TMDB
    settings.append(("TMDB", "TMDB_API_KEY", "********" if getattr(config, "tmdb_api_key", None) else None))

    # Filtering
    if getattr(config, "filter", False):
        settings.append(("Filtering", "FILTER", "True"))
        if getattr(config, "type", None) is not None:
            settings.append(("Filtering", "TYPE", str(config.type)))
        if getattr(config, "year", None) is not None:
            settings.append(("Filtering", "YEAR", str(config.year)))
        if getattr(config, "contains", None) is not None:
            settings.append(("Filtering", "CONTAINS", str(config.contains)))
        if getattr(config, "id", None) is not None:
            settings.append(("Filtering", "ID", str(config.id)))

    # Paths (optional/for debugging)
    if hasattr(config, "script_dir"):
        settings.append(("Paths", "SCRIPT_DIR", str(config.script_dir)))
    if hasattr(config, "log_dir"):
        settings.append(("Paths", "LOG_DIR", str(config.log_dir)))

    console.print("[bold blue]🔧 Current Settings[/bold blue]\n")
    for section, setting, value in settings:
        table.add_row(section, setting, value)

    if config.log_level == "DEBUG":
        section_width = max(len(str(s[0])) for s in settings + [("Section", "", "")])
        setting_width = max(len(str(s[1])) for s in settings + [("", "Setting", "")])
        value_width = max(len(str(s[2])) if s[2] is not None else 0 for s in settings + [("", "", "Value")])
        header = f"{'Section'.ljust(section_width)} | {'Setting'.ljust(setting_width)} | {'Value'.ljust(value_width)}"
        sep = "-" * len(header)
        log.debug("🔧 Current Settings")
        log.debug(header)
        log.debug(sep)
        for section, setting, value in settings:
            val_str = str(value) if value is not None else ""
            log.debug(
                f"{section.ljust(section_width)} | {setting.ljust(setting_width)} | {val_str.ljust(value_width)}"
            )
    console.print(table)


def perform_revert(config: "IdarrConfig", items: list["MediaItem"]) -> bool:
    """
    Revert renamed files to their original state using the cache's rename_history.
    Args:
        config: The configuration namespace (must contain cache and source_dir).
        items: List of MediaItem objects to revert.
    Returns:
        True if operation completes (even with errors), False otherwise.
    """
    if not config.revert:
        return False
    reverted = 0
    for item in items:
        cache_entry = config.cache.get(config.cache_manager.get_cache_key(item))
        if not cache_entry:
            log.warning(f"⚠️ No cache entry for '{item.title}' ({item.year})")
            continue
        rhist = cache_entry.get("rename_history", [])
        if not rhist:
            log.warning(f"⚠️ No rename history for '{item.title}' ({item.year})")
            continue

        for entry in reversed(rhist):
            to_name = entry.get("to")
            from_name = entry.get("from")
            if not to_name or not from_name:
                continue
            to_path = os.path.join(config.source_dir, to_name)
            from_path = os.path.join(config.source_dir, from_name)
            if os.path.exists(to_path):
                try:
                    os.rename(to_path, from_path)
                    log.info(f"↩️ Reverted: {to_name} → {from_name}", "YELLOW")
                    reverted += 1
                    cf = set(cache_entry.get("current_filenames", []))
                    if to_name in cf:
                        cf.remove(to_name)
                    cf.add(from_name)
                    cache_entry["current_filenames"] = sorted(cf)
                except Exception as e:
                    log.error(f"Failed to revert {to_name}: {e}")
            else:
                log.warning(f"⚠️ File not found for revert: {to_path}")

        cache_entry["rename_history"] = []

    log.info(f"↩️ Reverted {reverted} file(s) using cache rename history.", "YELLOW")
    return True


def filter_items(args: argparse.Namespace, items: list["MediaItem"]) -> list["MediaItem"]:
    if getattr(args, "filter", False):
        if args.type:
            items = [i for i in items if i.type == args.type]
        if args.year:
            items = [i for i in items if i.year == args.year]
        if args.contains:
            items = [i for i in items if args.contains.lower() in i.title.lower()]
        if args.id:
            prefix = args.id.lower().split("-")[0]
            id_value = args.id[len(prefix) + 1 :]
            if prefix == "tmdb":
                items = [i for i in items if str(i.tmdb_id) == id_value]
            elif prefix == "tvdb":
                items = [i for i in items if str(i.tvdb_id) == id_value]
            elif prefix == "imdb":
                items = [i for i in items if str(i.imdb_id) == id_value]
            else:
                log.error(f"❌ Invalid --id format. Use tmdb-123, tvdb-456, or imdb-tt1234567")
                exit(1)
    if args.limit:
        items = items[: args.limit]
    return items


def export_csvs(
    updated_items: list["MediaItem"], file_updates: list[Any], duplicate_log: list[dict[str, Any]]
) -> None:
    """
    Export all CSVs (updated_files, unmatched_cases, duplicates_log, tvdb_missing_cases) with shared logic.
    Args:
        updated_items: List of updated MediaItems.
        file_updates: List of file update tuples.
        duplicate_log: List of duplicate log dictionaries.
    Returns:
        None
    """

    with console.status("[cyan]Exporting CSVs... Please wait.", spinner="dots"):

        def write_csv(path, rows, fieldnames):
            if not rows:
                return
            with open(path, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(
                    csvfile,
                    fieldnames=fieldnames,
                    quoting=csv.QUOTE_ALL,
                )
                writer.writeheader()
                for row in rows:
                    writer.writerow(row)
            if rows:
                log.info(f"⚙️ CSV written to {path}")

        updated_files_path = os.path.join(LOG_DIR, "updated_files.csv")
        updated_files_fieldnames = [
            "media_type",
            "original_filename",
            "new_filename",
            "original_title",
            "new_title",
            "original_year",
            "new_year",
            "tmdb_id",
            "new_tmdb_id",
            "tvdb_id",
            "new_tvdb_id",
            "imdb_id",
            "new_imdb_id",
            "match_reason",
        ]
        updated_rows = []
        for media_type, old_fn, new_fn in file_updates:
            matched = next(
                (
                    item
                    for item in updated_items
                    if item.type == media_type and any(os.path.basename(fp) == old_fn for fp in item.files)
                ),
                None,
            )
            row = {
                "media_type": media_type,
                "original_filename": os.path.basename(old_fn),
                "new_filename": os.path.basename(new_fn),
                "original_title": getattr(matched, "title", ""),
                "new_title": getattr(matched, "new_title", ""),
                "original_year": getattr(matched, "year", ""),
                "new_year": getattr(matched, "new_year", ""),
                "tmdb_id": getattr(matched, "tmdb_id", ""),
                "new_tmdb_id": getattr(matched, "new_tmdb_id", ""),
                "tvdb_id": getattr(matched, "tvdb_id", ""),
                "new_tvdb_id": getattr(matched, "new_tvdb_id", ""),
                "imdb_id": getattr(matched, "imdb_id", ""),
                "new_imdb_id": getattr(matched, "new_imdb_id", ""),
                "match_reason": getattr(matched, "match_reason", ""),
            }
            updated_rows.append(row)
        write_csv(updated_files_path, updated_rows, updated_files_fieldnames)

        unmatched_cases_path = os.path.join(LOG_DIR, "unmatched_cases.csv")
        unmatched_fieldnames = [
            "files",
            "title",
            "year",
            "media_type",
            "tmdb_id",
            "imdb_id",
            "tvdb_id",
            "match_reason",
        ]

        all_keys = set()
        for case in UNMATCHED_CASES:
            all_keys.update(case.keys())
        unmatched_extra = [k for k in sorted(all_keys) if k not in unmatched_fieldnames]
        final_unmatched_fieldnames = unmatched_fieldnames + unmatched_extra

        unmatched_rows = []
        for case in UNMATCHED_CASES:
            row = dict(case)
            if "files" in row:
                files = row["files"]
                if isinstance(files, str):
                    files = [f for f in files.split(";") if f]
                elif not isinstance(files, list):
                    files = []
                row["files"] = ";".join(os.path.basename(f) for f in files)
            unmatched_rows.append(row)
        write_csv(unmatched_cases_path, unmatched_rows, final_unmatched_fieldnames)

        duplicates_csv_path = os.path.join(LOG_DIR, "duplicates_log.csv")
        duplicate_fieldnames = [
            "action",
            "kept_file",
            "kept_path",
            "kept_ctime",
            "moved_file",
            "moved_path",
            "moved_ctime",
        ]

        all_dupe_keys = set()
        for entry in duplicate_log:
            all_dupe_keys.update(entry.keys())
        extra_dupe = [k for k in sorted(all_dupe_keys) if k not in duplicate_fieldnames]
        final_duplicate_fieldnames = duplicate_fieldnames + extra_dupe

        write_csv(duplicates_csv_path, duplicate_log, final_duplicate_fieldnames)

        tvdb_missing_path = os.path.join(LOG_DIR, "tvdb_missing_cases.csv")
        tvdb_missing_fieldnames = ["title", "year", "tmdb_id", "imdb_id", "files"]
        all_tvdb_keys = set()
        for entry in TVDB_MISSING_CASES:
            all_tvdb_keys.update(entry.keys())
        extra_tvdb = [k for k in sorted(all_tvdb_keys) if k not in tvdb_missing_fieldnames]
        final_tvdb_fieldnames = tvdb_missing_fieldnames + extra_tvdb

        tvdb_rows = []
        for entry in TVDB_MISSING_CASES:
            row = dict(entry)
            if "files" in row:
                files = row["files"]
                if isinstance(files, str):
                    files = [f for f in files.split(";") if f]
                elif not isinstance(files, list):
                    files = []
                row["files"] = ";".join(os.path.basename(f) for f in files)
            tvdb_rows.append(row)
        write_csv(tvdb_missing_path, tvdb_rows, final_tvdb_fieldnames)


def load_ignore_and_sync_pending(config: "IdarrConfig") -> tuple[set[str], dict[str, Any]]:
    """
    Keep ignored_titles.jsonc and pending_matches.jsonc in sync:
    - If a title is in ignored_titles, remove from pending_matches and mark as ignored in cache.
    - If a title is removed from ignored_titles, re-populate in pending_matches (unless already matched/found in cache) and un-ignore in cache.
    """
    ignore_file = getattr(config, "ignore_file", None) or os.path.join(LOG_DIR, "ignored_titles.jsonc")
    pending_file = PENDING_MATCHES_PATH

    # 1. Load ignore list
    ignored_title_keys = set()
    ignore_file_header_lines = []
    if os.path.exists(ignore_file):
        with open(ignore_file, "r", encoding="utf-8") as f:
            lines = []
            for line in f:
                stripped = line.lstrip()
                if stripped.startswith("//") or stripped.startswith("#"):
                    ignore_file_header_lines.append(line)
                    continue
                lines.append(line)
            try:
                data = json.loads("".join(lines))
                if isinstance(data, dict):
                    ignored_title_keys = set(data.keys())
                elif isinstance(data, list):
                    ignored_title_keys = set(data)
            except Exception as e:
                log.warning(f"⚠️ Failed to load ignore file {ignore_file}: {e}")

    # 2. Load pending matches
    pending_matches = {}
    if os.path.exists(pending_file):
        with open(pending_file, encoding="utf-8") as f:
            lines = [line for line in f if not line.lstrip().startswith("//")]
            try:
                pending_matches = json.loads("".join(lines))
            except Exception as e:
                log.warning(f"⚠️ Failed to load pending matches: {e}")
                pending_matches = {}

    # 3. Handle ignores -> remove from pending and set ignored in cache
    for ignored_key in list(ignored_title_keys):
        # Remove from pending if present
        if ignored_key in pending_matches:
            del pending_matches[ignored_key]
            log.info(f"🔄 Removed '{ignored_key}' from pending_matches due to ignore list.")

        # Set as ignored in cache
        m = re.match(TITLE_YEAR_REGEX, ignored_key)
        title = m.group(1)
        year = int(m.group(2)) if m and m.group(2) else None
        for cache_key, entry in config.cache.items():
            if entry.get("title") == title and (
                entry.get("year") == year or (year is None and not entry.get("year"))
            ):
                if entry.get("status") != "ignored":
                    entry["status"] = "ignored"
                    config.cache_manager.upsert(
                        cache_key,
                        {
                            **entry,
                            "status": "ignored",
                            "last_checked": datetime.now().isoformat(),
                        },
                    )

    # 4. Handle removed from ignore list: should be re-added to pending if not found in cache
    # (We compare cache for ignored items that are no longer ignored, and revert their status, add to pending)
    previously_ignored = set()
    if os.path.exists(config.cache_manager.db_path):
        for cache_key, entry in config.cache.items():
            if entry.get("status") == "ignored":
                label = entry.get("title")
                year = entry.get("year")
                if year:
                    key = f"{label} ({year})"
                else:
                    key = f"{label}"
                previously_ignored.add(key)

    for was_ignored in previously_ignored:
        if was_ignored not in ignored_title_keys:
            # Remove "ignored" status from cache
            m = re.match(TITLE_YEAR_REGEX, was_ignored)
            title = m.group(1)
            year = int(m.group(2)) if m and m.group(2) else None
            for cache_key, entry in config.cache.items():
                if entry.get("title") == title and (
                    entry.get("year") == year or (year is None and not entry.get("year"))
                ):
                    # If item is not "found" or "matched" (i.e., not resolved), put back to pending
                    if entry.get("status") == "ignored":
                        entry["status"] = "unknown"
                        config.cache_manager.upsert(
                            cache_key,
                            {
                                **entry,
                                "status": "unknown",
                                "last_checked": datetime.now().isoformat(),
                            },
                        )
                    # Add to pending_matches if not already matched/found
                    if was_ignored not in pending_matches:
                        pending_matches[was_ignored] = "add_tmdb_url_here"
                        log.info(
                            f"🔄 Restored '{was_ignored}' to pending_matches (removed from ignore list)."
                        )

    save_pending_matches(pending_matches, pending_file)

    return ignored_title_keys, pending_matches


def resolve_pending_matches(config):
    """
    Resolve pending matches by updating cache with user-supplied TMDB URLs/IDs.
    - Always uses the canonical TMDB title/type for the match, not the original pending key.
    - Logs retitling if the TMDB title differs from the pending match key.
    - Removes the old entry using the original key from the cache after upsert.
    - Only supports simple "Old Base Name": "tmdb_url" mapping.
    Side effects:
        Modifies cache, writes to IGNORE and PENDING files, updates SQLite.
    """
    ignored = set()
    existing = []
    if os.path.exists(IGNORED_TITLES_PATH):
        with open(IGNORED_TITLES_PATH, encoding="utf-8") as f:
            lines = f.readlines()
        content_lines = [line for line in lines if not line.lstrip().startswith("//")]
        try:
            data = json.loads("".join(content_lines))
            if isinstance(data, list):
                existing = data
                ignored.update(existing)
        except (json.JSONDecodeError, ValueError):
            existing = []
    config.ignored_title_keys = set(existing)

    if not os.path.exists(PENDING_MATCHES_PATH):
        return

    with open(PENDING_MATCHES_PATH, encoding="utf-8") as f:
        lines = f.readlines()
    content_lines = [line for line in lines if not line.lstrip().startswith("//")]
    try:
        pending = json.loads("".join(content_lines))
    except (json.JSONDecodeError, ValueError):
        pending = {}

    updated = False

    for key, value in list(pending.items()):
        if value == "ignore":
            m = re.match(TITLE_YEAR_REGEX, key)
            title = m.group(1)
            year = int(m.group(2)) if m and m.group(2) else None
            for cache_key, entry in config.cache.items():
                if entry.get("title") == title and (
                    entry.get("year") == year or (year is None and not entry.get("year"))
                ):
                    config.cache_manager.upsert(
                        cache_key,
                        {
                            "title": title,
                            "year": year,
                            "type": entry.get("type"),
                            "tmdb_id": entry.get("tmdb_id"),
                            "tvdb_id": entry.get("tvdb_id"),
                            "imdb_id": entry.get("imdb_id"),
                            "status": "ignored",
                            "last_checked": datetime.now().isoformat(),
                        },
                    )
                    break
            ignored.add(key)
            del pending[key]
            continue
        if value and value != "add_tmdb_url_here":
            # Parse TMDB type and ID from URL if present
            pending_type = None
            tmdb_id = None
            if isinstance(value, str) and "themoviedb.org" in value:
                # Determine type from URL
                if "/tv/" in value:
                    pending_type = "tv_series"
                elif "/movie/" in value:
                    pending_type = "movie"
                elif "/collection/" in value:
                    pending_type = "collection"
                # Extract TMDB ID
                m = re.search(r"/(tv|movie|collection)/(\d+)", value)
                if m:
                    tmdb_id = int(m.group(2))
            else:
                # Fall back to previous behavior
                m = re.search(r"(\d+)", value) if isinstance(value, str) else None
                if m:
                    tmdb_id = int(m.group(1))
            if not tmdb_id:
                del pending[key]
                continue

            m = re.match(TITLE_YEAR_REGEX, key)
            title = m.group(1)
            year = int(m.group(2)) if m and m.group(2) else None

            old_key = None
            entry = None
            for cache_key, cache_entry in config.cache.items():
                if cache_entry.get("title") == title and (
                    cache_entry.get("year") == year or (year is None and not cache_entry.get("year"))
                ):
                    old_key = cache_key
                    entry = cache_entry
                    break
            if not entry:
                del pending[key]
                continue

            # Use TMDB canonical title/type for the match
            media_item = MediaItem(
                config=config,
                type=pending_type if pending_type else entry.get("type", "movie"),
                title=entry.get("title"),
                year=entry.get("year"),
                tmdb_id=tmdb_id,
                tvdb_id=entry.get("tvdb_id"),
                imdb_id=entry.get("imdb_id"),
                files=entry.get("current_filenames", []),
            )
            media_item.enrich()
            # Always use canonical TMDB title/year/type for the updated entry
            resolved_title = media_item.new_title or media_item.title
            resolved_year = media_item.new_year if media_item.new_year is not None else media_item.year
            resolved_tmdb_id = (
                media_item.new_tmdb_id if media_item.new_tmdb_id is not None else media_item.tmdb_id
            )
            resolved_type = media_item.type
            new_key = config.cache_manager.get_cache_key(
                MediaItem(
                    config=config,
                    type=resolved_type,
                    title=resolved_title,
                    year=resolved_year,
                    tmdb_id=resolved_tmdb_id,
                    tvdb_id=media_item.new_tvdb_id or media_item.tvdb_id,
                    imdb_id=media_item.new_imdb_id or media_item.imdb_id,
                    files=media_item.files,
                )
            )

            original_title = media_item.title
            new_title = media_item.new_title

            if new_title and new_title != original_title:
                log.info(f"🔄 Retitling '{original_title}' → '{new_title}' (from TMDB metadata)")

            config.cache_manager.upsert(
                new_key,
                {
                    "title": resolved_title,
                    "year": resolved_year,
                    "type": resolved_type,
                    "tmdb_id": resolved_tmdb_id,
                    "tvdb_id": media_item.new_tvdb_id or media_item.tvdb_id,
                    "imdb_id": media_item.new_imdb_id or media_item.imdb_id,
                    "status": "found",
                    "last_checked": datetime.now().isoformat(),
                },
            )
            # Remove the old entry using the original key from the cache after upsert
            if old_key in config.cache_manager.cache:
                config.cache_manager.cache.pop(old_key)
            with sqlite3.connect(config.cache_manager.db_path) as conn:
                conn.execute("DELETE FROM cache WHERE key = ?", (old_key,))
                conn.commit()

            updated = True
            del pending[key]

    header_lines = []
    if os.path.exists(IGNORED_TITLES_PATH):
        with open(IGNORED_TITLES_PATH, encoding="utf-8") as f_in:
            for line in f_in:
                if line.lstrip().startswith("//"):
                    header_lines.append(line.rstrip("\n"))
                else:
                    break
    with open(IGNORED_TITLES_PATH, "w", encoding="utf-8") as f_out:
        for comment in header_lines:
            f_out.write(comment + "\n")
        json.dump(sorted(ignored), f_out, indent=2, ensure_ascii=False)
        f_out.write("\n")

    if updated:
        config.cache = config.cache_manager.cache
    save_pending_matches(pending, PENDING_MATCHES_PATH)


def summarize_run(
    start_time: float,
    items: list["MediaItem"],
    updated_items: list["MediaItem"],
    file_updates: list[Any],
    config: "IdarrConfig",
) -> None:
    """
    Summarize the run and print/log summary statistics.
    Args:
        start_time: Start time as a float (from time.time()).
        items: List of input MediaItems.
        updated_items: List of updated MediaItems.
        file_updates: List of file update tuples.
        config: Configuration object.
    Returns:
        None
    """
    elapsed_seconds = int(time.time() - start_time)
    if elapsed_seconds < 60:
        elapsed_str = f"{elapsed_seconds}s"
    elif elapsed_seconds < 3600:
        mins, secs = divmod(elapsed_seconds, 60)
        elapsed_str = f"{mins}m {secs}s"
    else:
        hours, rem = divmod(elapsed_seconds, 3600)
        mins, secs = divmod(rem, 60)
        elapsed_str = f"{hours}h {mins}m {secs}s"

    cache_skipped = sum(1 for item in items if getattr(item, "skipped_by_cache", False))

    ambiguous_count = sum(1 for case in UNMATCHED_CASES if case.get("match_reason") == "ambiguous")

    labels = [
        ("⏱️ Elapsed Time", elapsed_str),
        ("📦 Items Processed", len(items)),
        ("✏️ Files Renamed", len(file_updates)),
        ("❌ Unmatched Items", len(UNMATCHED_CASES)),
        ("🤷 Ambiguous Matches", ambiguous_count),
        ("📺 TVDB Missing (TV)", len(TVDB_MISSING_CASES)),
        ("🔁 Reclassified (TV)", len(RECLASSIFIED)),
        ("💾 Cache Skipped", cache_skipped),
        ("📡 TMDB API Calls", getattr(config, "_api_calls", 0)),
    ]

    table = Table(show_header=False, box=None, padding=(0, 1))
    for label, value in labels:
        table.add_row(label, str(value))
    console.rule("[bold]Summary Report")
    console.print(table)
    console.rule()
    log.info("Summary Report:", color="", console=False)
    for label, value in labels:
        log.info(f"{label}: {value}", color="", console=False)


def handle_special_arguments(args, config, ignored_title_keys):
    if getattr(args, "prune", False):
        prune_orphaned_cache_entries(config)
        sys.exit(0)
    if getattr(args, "purge", False):
        deleted = config.cache_manager.delete(args.purge)
        if deleted:
            print(f"Cache entry/entries matching '{args.purge}' deleted.")
        else:
            print(f"No cache entry found matching '{args.purge}'.")
        sys.exit(0)
    if getattr(args, "pending_matches", False):
        resolve_pending_matches(config)
        just_resolved_items = []
        source_dir = (
            getattr(config, "source_dir", None)
            or getattr(config, "source", None)
            or getattr(getattr(config, "args", {}), "source", None)
        )
        for cache_key, entry in config.cache.items():
            if entry.get("status") == "found" and cache_key not in ignored_title_keys:
                media_item = MediaItem(
                    config=config,
                    type=entry.get("type", "movie"),
                    title=entry.get("title"),
                    year=entry.get("year"),
                    tmdb_id=entry.get("tmdb_id"),
                    tvdb_id=entry.get("tvdb_id"),
                    imdb_id=entry.get("imdb_id"),
                    files=[
                        f if os.path.isabs(f) else os.path.join(source_dir, f)
                        for f in entry.get("current_filenames", [])
                    ],
                )
                just_resolved_items.append(media_item)
        if just_resolved_items:
            rename_files(just_resolved_items, config)
        else:
            log.info("No newly resolved pending matches to process.")
        sys.exit(0)
        sys.exit(0)
    if getattr(args, "revert", False):
        items = scan_files_in_flat_folder(config)
        items = filter_items(args, items)
        if items:
            perform_revert(config, items)
        sys.exit(0)


def main():
    global log
    args = parse_args()
    log = LogManager(__name__, LOG_PATH)
    log.info(
        f"************************* IDARR Version: {FULL_VERSION} *************************",
        color="",
        console=False,
    )
    if getattr(args, "filter", False):
        if not (args.type or args.year or args.contains):
            log.error("❌ --filter requires at least one of --type, --year, or --contains")
            exit(1)
    config = load_runtime_config(args)
    ignored_title_keys, pending_matches = load_ignore_and_sync_pending(config)
    config.pending_matches = pending_matches
    config.ignored_title_keys = ignored_title_keys

    handle_special_arguments(args, config, ignored_title_keys)
    resolve_pending_matches(config)

    if config.log_level == "DEBUG":
        print_settings(config)
    start_time = time.time()
    items = scan_files_in_flat_folder(config)
    if any(entry.get("type") == "tv_series" and not entry.get("tvdb_id") for entry in config.cache.values()):
        switch = config.tmdb_query_service.rehydrate_missing_tvdb_ids(
            config.cache, max_age_days=config.tvdb_frequency
        )
        if switch:
            prune_orphaned_cache_entries(config)
    items = filter_items(args, items)
    if not items:
        log.info("No items found to process. Exiting.")
        return
    updated_items = handle_data(config, items)
    file_updates, duplicate_log = rename_files(updated_items, config)
    export_csvs(updated_items, file_updates, duplicate_log)
    summarize_run(start_time, items, updated_items, file_updates, config)


if __name__ == "__main__":
    try:
        load_dotenv(override=True)
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user (Ctrl+C). Exiting gracefully.")
    except Exception as e:
        console.print("[bold red]💥 Unexpected error:[/bold red]", e)
        console.print(Traceback(show_locals=False))
        sys.exit(1)
