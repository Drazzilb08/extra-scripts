version = "1.3.0"

import sys
import os
import re
import csv
import logging
import unicodedata
import string
import time
import argparse
import shutil
from typing import Pattern, Optional, Any
from collections import defaultdict
from difflib import SequenceMatcher
from types import SimpleNamespace
from functools import wraps
import json
from datetime import datetime, timedelta

if sys.version_info < (3, 9):
    print("Python 3.9 or higher is required. Detected version: {}.{}.{}".format(*sys.version_info[:3]))
    exit(1)

try:
    from tmdbapis import TMDbAPIs
    from ratelimit import limits, RateLimitException
    from tqdm import tqdm
    from unidecode import unidecode
    from dotenv import load_dotenv
    from rich.console import Console
    from rich.table import Table
except ImportError as e:
    missing = getattr(e, "name", None) or str(e)
    print(f"❌ Missing dependency: {missing}. Please install all dependencies with 'pip install -r requirements.txt'.")
    exit(1)

load_dotenv(override=True)

try:
    import subprocess

    BUILD_NUMBER = subprocess.check_output(["git", "rev-list", "--count", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    FULL_VERSION = f"{version}.build{BUILD_NUMBER}"
except Exception:
    FULL_VERSION = version

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(SCRIPT_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
log_path = os.path.join(LOG_DIR, "idarr.log")
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
log_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
YEAR_REGEX: Pattern = re.compile(r"\s?\((\d{4})\)(?!.*Collection).*")
SEASON_PATTERN: Pattern = re.compile(r"(?:\s*-\s*Season\s*\d+|_Season\d{1,2}|\s*-\s*Specials|_Specials)", re.IGNORECASE)
TMDB_ID_REGEX: Pattern = re.compile(r"tmdb[-_\s](\d+)")
TVDB_ID_REGEX: Pattern = re.compile(r"tvdb[-_\s](\d+)")
IMDB_ID_REGEX: Pattern = re.compile(r"imdb[-_\s](tt\d+)")
UNMATCHED_CASES: list[dict[str, Any]] = []
TVDB_MISSING_CASES: list[dict[str, Any]] = []
RECLASSIFIED: list[dict[str, Any]] = []
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp"]
CACHE_PATH = os.path.join(SCRIPT_DIR, "cache", "idarr_cache.json")
PENDING_MATCHES_PATH = os.path.join(LOG_DIR, "pending_matches.json")


def parse_tmdb_url(url: str):
    """
    Given a TMDB URL, returns (tmdb_id, type) tuple, or (None, None) if not parsable.
    Example: https://www.themoviedb.org/movie/586595-after-midnight -> (586595, "movie")
    """
    m = re.match(r'https?://www\.themoviedb\.org/(movie|tv|collection)/(\d+)', url)
    if not m:
        return None, None
    tmap = {"movie": "movie", "tv": "tv_series", "collection": "collection"}
    return int(m.group(2)), tmap.get(m.group(1))

# Pending matches utilities
def load_pending_matches() -> dict[str, int | None]:
    if os.path.exists(PENDING_MATCHES_PATH):
        with open(PENDING_MATCHES_PATH, "r") as f:
            return json.load(f)
    return {}

def save_pending_matches(pending: dict[str, int | None]) -> None:
    os.makedirs(os.path.dirname(PENDING_MATCHES_PATH), exist_ok=True)
    with open(PENDING_MATCHES_PATH, "w") as f:
        json.dump(pending, f, indent=2)


def load_cache() -> dict[str, Any]:
    """
    Load the cache from the cache file if it exists.
    Returns:
        A dictionary representing the cache.
    """
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "r") as f:
            return json.load(f)
    return {}


def save_cache(cache: dict[str, Any], active_keys: set[str]) -> None:
    """
    Save the cache to the cache file, filtering to only include active keys.
    Args:
        cache: The current cache dictionary.
        active_keys: Set of keys to retain in the cache.
    """
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    cache = {k: v for k, v in cache.items() if k in active_keys}
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)


def is_recent(last_checked: str) -> bool:
    """
    Determine if a cached entry is recent, based on FREQUENCY_DAYS.
    Args:
        last_checked: ISO format date string.
    Returns:
        True if last_checked is within FREQUENCY_DAYS, else False.
    """
    freq_days = globals().get("FREQUENCY_DAYS", 30)
    try:
        last_time = datetime.fromisoformat(last_checked)
        return datetime.now() - last_time < timedelta(days=freq_days)
    except Exception:
        return False


def console(msg: str, level: str = "WHITE") -> None:
    """
    Print a message to the console with optional color, if not in QUIET mode.
    Args:
        msg: Message string to print.
        level: Color level as a string.
    """
    colors = {
        "WHITE": "\033[97m",
        "YELLOW": "\033[93m",
        "RED": "\033[91m",
        "BLUE": "\033[94m",
        "GREEN": "\033[92m",
    }
    end = "\033[0m"
    if not globals().get("QUIET", False):
        print(f"{colors.get(level,'')}{msg}{end}")


def get_cache_key(item) -> str:
    tmdb = getattr(item, "new_tmdb_id", None) or getattr(item, "tmdb_id", None)
    tvdb = getattr(item, "new_tvdb_id", None) or getattr(item, "tvdb_id", None)
    imdb = getattr(item, "new_imdb_id", None) or getattr(item, "imdb_id", None)
    if tmdb or tvdb or imdb:
        return f"{tmdb or 'no-tmdb'}-{tvdb or 'no-tvdb'}-{imdb or 'no-imdb'}-{item.type}"
    normalized_title = normalize_with_aliases(item.title)
    return f"{normalized_title}-{item.year or 'noyear'}-{item.type}"


class NoResultsError(Exception):
    """Custom exception for TMDB lookups that return no results."""

    pass


class MediaItem:
    """
    Represents a media item (movie, tv_series, or collection) with associated metadata and files.
    """

    def __init__(
        self,
        config: SimpleNamespace,
        type: str,
        title: str,
        year: Optional[int],
        tmdb_id: Optional[int],
        tvdb_id: Optional[int] = None,
        imdb_id: Optional[str] = None,
        files: Optional[list[str]] = None,
    ) -> None:
        self.type = type
        self.title = title
        self.year = year
        self.tmdb_id = tmdb_id
        self.tvdb_id = tvdb_id
        self.imdb_id = imdb_id
        self.files = files or []
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
        self.config = config
        self.renamed: bool = False

    def enrich(self) -> bool:
        """
        Perform TMDB lookup and update self.* fields with enriched metadata.
        Uses cache if available and recent.
        Returns:
            True if enrichment succeeded, False otherwise.
        """
        if not hasattr(self.config, "_api_calls"):
            self.config._api_calls = 0
        cache_key = get_cache_key(self)
        cached = self.config.cache.get(cache_key)
        should_skip = not self.config.dry_run and cached and is_recent(cached.get("last_checked", ""))
        self.skipped_by_cache = should_skip

        if not self.config.quiet and not should_skip:
            console("")
            logger.info("")

        if should_skip:
            if cached.get("no_result"):
                logger.info(f"📦 Skipping (previously not found): {self.title} ({self.year})")
                return False
            self.new_title = cached.get("title")
            self.new_year = cached.get("year")
            self.new_tmdb_id = cached.get("tmdb_id")
            self.new_tvdb_id = cached.get("tvdb_id")
            self.new_imdb_id = cached.get("imdb_id")
            logger.info(f"📦 Used cached metadata for {self.new_title or self.title} ({self.new_year or self.year})")
            return True
        self.config._api_calls += 1
        result = query_tmdb(
            self,
            self.type,
        )
        if not result:
            self.config.cache[cache_key] = {
                "last_checked": datetime.now().isoformat(),
                "no_result": True,
            }
            self.match_failed = True
            return False

        if hasattr(result, "id"):
            if result.id and (self.tmdb_id is None or result.id != self.tmdb_id):
                if not self.config.quiet:
                    console(f"  ⚠️ TMDB ID mismatch: {self.tmdb_id} → {result.id}", "YELLOW")
                logger.warning(f"  ⚠️ TMDB ID mismatch: {self.tmdb_id} → {result.id}")
            self.new_tmdb_id = result.id
        if hasattr(result, "tvdb_id"):
            if getattr(result, "tvdb_id", None) and (self.tvdb_id is None or result.tvdb_id != self.tvdb_id):
                if not self.config.quiet:
                    console(
                        f"  ⚠️ TVDB ID mismatch: {self.tvdb_id} → {result.tvdb_id}",
                        "YELLOW",
                    )
                logger.warning(f"  ⚠️ TVDB ID mismatch: {self.tvdb_id} → {result.tvdb_id}")
            self.new_tvdb_id = result.tvdb_id
        if hasattr(result, "imdb_id"):
            # Always log when the original is None or changes
            if getattr(result, "imdb_id", None) and (self.imdb_id is None or result.imdb_id != self.imdb_id):
                if not self.config.quiet:
                    console(
                        f"  ⚠️ IMDB ID mismatch: {self.imdb_id} → {result.imdb_id}",
                        "YELLOW",
                    )
                logger.warning(f"  ⚠️ IMDB ID mismatch: {self.imdb_id} → {result.imdb_id}")
            self.new_imdb_id = result.imdb_id

        tmdb_title = getattr(result, "title", getattr(result, "name", None))
        tmdb_date = getattr(result, "first_air_date", getattr(result, "release_date", None))
        if tmdb_date:
            res_year = tmdb_date.year if hasattr(tmdb_date, "year") else int(str(tmdb_date)[:4])
            if res_year != self.year:
                self.new_year = res_year
        if tmdb_title and tmdb_title != self.title:
            self.new_title = tmdb_title

        self.config.cache[cache_key] = {
            "last_checked": datetime.now().isoformat(),
            "type": self.type,
            "tmdb_id": self.new_tmdb_id or self.tmdb_id,
            "tvdb_id": self.new_tvdb_id or self.tvdb_id,
            "imdb_id": self.new_imdb_id or self.imdb_id,
            "year": self.new_year or self.year,
            "title": self.new_title or self.title,
        }
        if not self.config.no_cache:
            save_cache(self.config.cache, set(self.config.cache.keys()))

        # Check if IDs were updated during enrichment and a better (ID-based) cache key is now available.
        new_cache_key = get_cache_key(self)
        if new_cache_key != cache_key:
            self.config.cache[new_cache_key] = self.config.cache.pop(cache_key)
            if not self.config.no_cache:
                save_cache(self.config.cache, set(self.config.cache.keys()))

        return True

    def needs_rename(self) -> bool:
        """
        Determine if any new_* fields exist, indicating a rename is needed.
        Returns:
            True if rename is needed, False otherwise.
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
        Does not perform I/O.
        Returns:
            List of tuples (old_filename, new_filename).
        """
        ops = []
        for file_path in self.files:
            directory, old = os.path.split(file_path)
            new = generate_new_filename(self, old)
            ops.append((old, new))
        return ops


def sleep_and_notify(func):
    """
    Decorator to catch RateLimitException and sleep for the required period before retrying.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        while True:
            try:
                return func(*args, **kwargs)
            except RateLimitException as e:
                console(
                    f"\033[93m[WARNING]\033[0m Rate limit hit, sleeping for {e.period_remaining:.2f} seconds",
                    "RED",
                )
                time.sleep(e.period_remaining)

    return wrapper


class RemoveColorFilter(logging.Filter):
    ansi_escape = re.compile(r"\x1B[@-_][0-?]*[ -/]*[@-~]")

    def filter(self, record):
        if isinstance(record.msg, str):
            record.msg = self.ansi_escape.sub("", record.msg)
        return True


file_handler = logging.FileHandler(log_path, mode="w")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(log_formatter)
file_handler.addFilter(RemoveColorFilter())
logger.addHandler(file_handler)


def normalize_str(string: str) -> str:
    """
    Normalize a string by removing diacritics, lowercasing, and collapsing whitespace.
    Preserves special characters, since they may be meaningful in media titles.
    Args:
        string: Input string.
    Returns:
        Normalized string.
    """
    nfkd = unicodedata.normalize("NFKD", string)
    only_ascii = nfkd.encode("ASCII", "ignore").decode()

    return re.sub(r"\s+", " ", only_ascii).lower().strip()


def normalize_with_aliases(string: str) -> str:
    """
    Normalize a string to ASCII, lowercased, with common abbreviations/aliases expanded.
    Returns a single canonical normalized form.
    """
    substitutions = [
        ("&", "and"),
        ("vs.", "versus"),
        ("vs", "versus"),
        ("ep.", "episode"),
        ("ep", "episode"),
        ("vol.", "volume"),
        ("vol", "volume"),
        ("pt.", "part"),
        ("pt", "part"),
        ("dr.", "doctor"),
        ("dr", "doctor"),
        ("+", "/"),
        ("_", ":"),
    ]
    nfkd = unicodedata.normalize("NFKD", string)
    string_ascii = nfkd.encode("ASCII", "ignore").decode()
    for a, b in substitutions:

        string_ascii = re.sub(rf"\b{re.escape(a)}\b", b, string_ascii, flags=re.IGNORECASE)
        string_ascii = re.sub(rf"\b{re.escape(b)}\b", a, string_ascii, flags=re.IGNORECASE)

    string_ascii = string_ascii.lower()
    string_ascii = re.sub(r"\s+", " ", string_ascii).strip()
    return string_ascii


def create_collection(title: str, tmdb_id: Optional[int], imdb_id: Optional[str], files: list[str]) -> dict[str, Any]:
    """
    Create a collection dictionary for MediaItem initialization.
    """
    return {
        "type": "collection",
        "title": title,
        "year": None,
        "imdb_id": imdb_id,
        "tmdb_id": tmdb_id,
        "files": files,
    }


def create_series(
    title: str,
    year: Optional[int],
    tvdb_id: Optional[int],
    imdb_id: Optional[str],
    tmdb_id: Optional[int],
    files: list[str],
) -> dict[str, Any]:
    """
    Create a tv_series dictionary for MediaItem initialization.
    """
    return {
        "type": "tv_series",
        "title": title,
        "year": year,
        "tvdb_id": tvdb_id,
        "imdb_id": imdb_id,
        "tmdb_id": tmdb_id,
        "files": files,
    }


def create_movie(
    title: str,
    year: Optional[int],
    tmdb_id: Optional[int],
    imdb_id: Optional[str],
    files: list[str],
) -> dict[str, Any]:
    """
    Create a movie dictionary for MediaItem initialization.
    """
    return {
        "type": "movie",
        "title": title,
        "year": year,
        "tmdb_id": tmdb_id,
        "imdb_id": imdb_id,
        "files": files,
    }


def extract_ids(text: str) -> tuple[Optional[int], Optional[int], Optional[str]]:
    """
    Extract TMDB, TVDB, and IMDB IDs from a string.
    Returns:
        Tuple of (tmdb_id, tvdb_id, imdb_id)
    """
    tmdb_match = TMDB_ID_REGEX.search(text)
    tmdb = int(tmdb_match.group(1)) if tmdb_match else None
    tvdb_match = TVDB_ID_REGEX.search(text)
    tvdb = int(tvdb_match.group(1)) if tvdb_match else None
    imdb_match = IMDB_ID_REGEX.search(text)
    imdb = imdb_match.group(1) if imdb_match else None
    return tmdb, tvdb, imdb


def parse_file_group(config: SimpleNamespace, base_name: str, files: list[str]) -> MediaItem:
    """
    Parse a group of files and return a MediaItem instance.
    Args:
        config: Configuration namespace.
        base_name: Base name representing the group.
        files: List of filenames in the group.
    Returns:
        MediaItem instance.
    """

    title = re.sub(r"{(tmdb|tvdb|imdb)-[^}]+}", "", base_name)
    title = re.sub(YEAR_REGEX, "", title).strip()
    year_match = YEAR_REGEX.search(base_name)
    year = int(year_match.group(1)) if year_match else None
    tmdb_id, tvdb_id, imdb_id = extract_ids(base_name)
    files = sorted([os.path.join(config.source_dir, file) for file in files if not file.startswith(".")])
    is_series = any(SEASON_PATTERN.search(file) for file in files)
    is_collection = not year

    # Determine type, preferring cached type if present
    cache_type = None
    tmp_media_item = None

    # Generate a temporary MediaItem to get the cache_key
    tmp_data = {
        "type": "movie",  # placeholder, will fix below
        "title": title,
        "year": year,
        "tmdb_id": tmdb_id,
        "tvdb_id": tvdb_id,
        "imdb_id": imdb_id,
        "files": files,
    }
    # If a TMDB ID is present, search all possible type cache keys
    cache_entry = None
    cache_type = None
    if tmdb_id is not None:
        for possible_type in ("movie", "tv_series", "collection"):
            tmp_data["type"] = possible_type
            tmp_media_item = MediaItem(**tmp_data, config=config)
            possible_cache = config.cache.get(get_cache_key(tmp_media_item))
            if possible_cache:
                cache_entry = possible_cache
                cache_type = possible_cache.get("type")
                break
    else:
        tmp_media_item = MediaItem(**tmp_data, config=config)
        cache_entry = config.cache.get(get_cache_key(tmp_media_item))
        if cache_entry:
            cache_type = cache_entry.get("type")

    if cache_type == "collection" or is_collection:
        data = create_collection(title, tmdb_id, imdb_id, files)
    elif cache_type == "tv_series" or is_series or tvdb_id:
        data = create_series(title, year, tvdb_id, imdb_id, tmdb_id, files)
    else:
        data = create_movie(title, year, tmdb_id, imdb_id, files)
    media_item = MediaItem(**data, config=config)
    if cache_entry:
        for field in ("tmdb_id", "tvdb_id", "imdb_id"):
            val = cache_entry.get(field)
            if val:
                setattr(media_item, field, val)
                new_field = f"new_{field}"
                if not getattr(media_item, new_field, None):
                    setattr(media_item, new_field, val)

    # Pending matches override
    title_key = f"{title} ({year})" if year else title
    pending_url = getattr(config, "pending_matches", {}).get(title_key)
    if pending_url and pending_url != "add_tmdb_url_here":
        tmdb_id, typ = parse_tmdb_url(pending_url)
        if not tmdb_id or not typ:
            raise ValueError(f"Invalid TMDB URL in pending_matches.json for '{title_key}': {pending_url}")
        media_item.tmdb_id = tmdb_id
        media_item.type = typ

    return media_item


def scan_files_in_flat_folder(config: SimpleNamespace) -> list[MediaItem]:
    """
    Scan a flat folder for image assets and group them into MediaItem instances.
    Args:
        config: Configuration namespace, must include source_dir attribute.
    Returns:
        List of MediaItem instances.
    """
    logger.info(f"📂 Scanning directory for image assets: {config.source_dir}")
    try:
        files = os.listdir(config.source_dir)
    except FileNotFoundError:
        return []
    groups = defaultdict(list)
    assets_dict = []
    for file in files:
        if file.startswith("."):
            continue
        ext = os.path.splitext(file)[-1].lower()
        if ext not in IMAGE_EXTENSIONS and config.remove_non_image_files:
            full_path = os.path.join(config.source_dir, file)
            if config.source_dir:
                if config.dry_run:
                    logger.info(f"[DRY RUN] Would delete non-image file: {file}")
                else:
                    try:
                        os.remove(full_path)
                        logger.info(f"🗑️ Removed non-image file: {file}")
                    except Exception as e:
                        logger.error(f"❌ Failed to delete {file}: {e}")

            continue
        title = file.rsplit(".", 1)[0]
        raw_title = SEASON_PATTERN.split(title)[0].strip()
        groups[raw_title].append(file)
    groups = dict(sorted(groups.items(), key=lambda x: x[0].lower()))

    all_files = [file for group in groups.values() for file in group if not file.startswith(".")]
    with tqdm(
        total=len(all_files),
        desc=f"Processing files {os.path.basename(config.source_dir)}",
        unit="file",
    ) as progress:
        for base_name, files in groups.items():
            assets_dict.append(parse_file_group(config, base_name, files))
            progress.update(len(files))
    total_assets = sum(len(v) for v in groups.values())
    logger.info(f"✅ Completed scanning: discovered {len(assets_dict)} asset groups covering {total_assets} files")

    return assets_dict


# ==== Helper functions for query_tmdb modularization ====


def fetch_by_tmdb_id(search: "MediaItem", media_type: str) -> Optional[Any]:
    """
    Fetch TMDB details for the given ID, checking all possible types.
    Returns enriched result or None if mismatch or no match.
    Sets search.match_reason appropriately.
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
                result = TMDB_CLIENT._api.movies_get_details(tmdb_id)
            elif m_type == "tv_series":
                result = TMDB_CLIENT._api.tv_get_details(tmdb_id)
            elif m_type == "collection":
                result = TMDB_CLIENT._api.collections_get_details(tmdb_id)

            if result and isinstance(result, dict):
                result = SimpleNamespace(**result)

            if result:
                res_id = getattr(result, "id", None)
                if res_id == tmdb_id:
                    # Check for exact title/year match
                    detail_title = getattr(result, "title", getattr(result, "name", getattr(result, "original_title", "")))
                    detail_year = None
                    date = getattr(result, "release_date", None) or getattr(result, "first_air_date", None)
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
                    candidates = fuzzy_match_candidates(
                        [result],
                        search,
                        strict=False,
                        ratio_threshold=0.5,
                        jaccard_threshold=0.0,
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


def fuzzy_match_candidates(
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
    If strict=True, return a single high-confidence match or None.
    If strict=False, return (candidates: list, scored: list of tuples).
    Parameters:
        search_results: List of candidate results to match.
        search: MediaItem to match against.
        strict: If True, requires high-confidence match using thresholds.
        ratio_threshold: Minimum SequenceMatcher ratio required for strict match (default 0.95).
        jaccard_threshold: Minimum word-level Jaccard similarity for strict match (default 0.85).
        year_tolerance: Allowed difference in years for strict match (default 0, requires exact year).
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
        y_score = 1.0 if res_year == search.year else 0.5 if res_year and search.year and abs(res_year - search.year) <= 1 else 0
        score = ratio * 2 + y_score
        scored.append((score, res))
        if strict:
            year_ok = (search.year is None and res_year is None) or (
                res_year is not None and search.year is not None and abs(res_year - search.year) <= year_tolerance
            )
            if ratio >= ratio_threshold and jaccard >= jaccard_threshold and year_ok:
                candidates.append(res)
            elif 0.0 <= ratio_threshold - ratio <= 0.05:
                msg = f"📉 Near-match: '{title}' (Ratio: {ratio:.3f}, Jaccard: {jaccard:.3f}, Year: {res_year})"
                logger.debug(msg)
        else:
            if score > 1.0:
                candidates.append(res)
    if strict:
        return candidates[0] if len(candidates) == 1 else None
    else:
        scored.sort(key=lambda x: x[0], reverse=True)
        return candidates


def exact_match_shortcut(search_results: list[Any], search: MediaItem) -> Optional[Any]:
    norm_search = normalize_with_aliases(search.title)
    for res in search_results:
        title = getattr(res, "title", getattr(res, "name", ""))
        if normalize_with_aliases(title) == norm_search:
            date = getattr(res, "release_date", getattr(res, "first_air_date", ""))
            year = date.year if hasattr(date, "year") else (int(date[:4]) if isinstance(date, str) and date[:4].isdigit() else None)
            if year == search.year:
                return res
    return None


def alternate_titles_fallback(search_results: list[Any], search: MediaItem, media_type: str) -> Optional[Any]:
    norm_search = normalize_with_aliases(search.title)
    for res in search_results:
        alt_list = getattr(res, "alternative_titles", [])
        for alt in alt_list:
            cand = alt.get("title") if isinstance(alt, dict) else str(alt)
            if normalize_with_aliases(cand) == norm_search:
                return res
    return None


def perform_tmdb_search(search: MediaItem, media_type: str) -> Optional[list[Any]]:
    if media_type == "collection":
        return TMDB_CLIENT.collection_search(query=search.title)
    elif media_type == "movie":
        return TMDB_CLIENT.movie_search(query=search.title, year=search.year)
    elif media_type == "tv_series":
        return TMDB_CLIENT.tv_search(query=search.title, first_air_date_year=search.year)
    else:
        msg = f"[SKIPPED] Unsupported media type '{media_type}' for '{search.title}'"
        logger.info(msg)
        return None


def match_by_id(search_results: list[Any], search: MediaItem, media_type: str) -> Optional[Any]:
    for res in search_results:
        if (
            (getattr(search, "tmdb_id", None) and getattr(res, "id", None) == search.tmdb_id)
            or (getattr(search, "tvdb_id", None) and getattr(res, "tvdb_id", None) == search.tvdb_id)
            or (getattr(search, "imdb_id", None) and getattr(res, "imdb_id", None) == search.imdb_id)
        ):
            return res
    return None


def match_by_original_title(search_results: list[Any], search: MediaItem, media_type: str) -> Optional[Any]:
    for res in search_results:
        orig_title = getattr(res, "original_title", None)
        if orig_title and normalize_with_aliases(orig_title) == normalize_with_aliases(search.title):
            return res
    return None


def _log_search_attempt(title, year, media_type, is_new_attempt):
    if is_new_attempt:
        msg = f"🔍 Searching TMDB for “{title}” ({year}) [{media_type}]..."
        logger.info(msg)
        console(msg)


def _log_result(header, msg, color="GREEN"):
    logger.info(header)
    logger.info(msg)
    console(header)
    console(msg, color)


def _try_id_lookup(search, media_type):
    result = fetch_by_tmdb_id(search, media_type)
    if result:
        reason = search.match_reason or "id"
        return result, reason
    return None, None


def _try_main_match(search_results, search, media_type):
    # 1. ID match
    id_match = match_by_id(search_results, search, media_type)
    if id_match:
        return id_match, "id"
    # 2. Ambiguous check (same title+year)
    exact_title = search.title.strip().lower()
    same_title_year = []
    for res in search_results:
        title = getattr(res, "title", getattr(res, "name", "")).strip().lower()
        date = getattr(res, "release_date", getattr(res, "first_air_date", ""))
        year_val = date.year if hasattr(date, "year") else (int(date[:4]) if isinstance(date, str) and date[:4].isdigit() else None)
        if title == exact_title and year_val == search.year:
            same_title_year.append(res)
    if len(same_title_year) > 1:
        search.match_failed = True
        search.match_reason = "ambiguous"
        return None, "ambiguous"
    # 3. Exact shortcut
    shortcut = exact_match_shortcut(search_results, search)
    if shortcut:
        return shortcut, "exact"
    # 4. Original title
    orig_match = match_by_original_title(search_results, search, media_type)
    if orig_match:
        return orig_match, "original"
    # 5. Alternate-title
    alt = alternate_titles_fallback(search_results, search, media_type)
    if alt:
        return alt, "alternate"
    # 6. Fuzzy
    fuzzy = fuzzy_match_candidates(search_results, search, strict=True)
    if fuzzy:
        return fuzzy, "fuzzy_norm"
    # 6. Fuzzy - wider year diff if search_results is 1 item
    if len(search_results) == 1:
        fuzzy = fuzzy_match_candidates(
            search_results, search, strict=True, ratio_threshold=0.9, jaccard_threshold=0.85, year_tolerance=2
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
        fuzzy_alt_result = fuzzy_match_candidates(alt_candidates, search, strict=True)
        if fuzzy_alt_result:
            fuzzy_alt = getattr(fuzzy_alt_result, "_orig_res", None)
    if fuzzy_alt:
        return fuzzy_alt, "fuzzy_alternate"
    return None, None


def _try_transformations(search_results, search, media_type):
    transformations = [
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
                # If we have no search results, try a new TMDB search with the transformed title
                try:
                    if not search_results:
                        alt_results = perform_tmdb_search(temp_search, media_type) or []
                    else:
                        alt_results = search_results
                except Exception as e:
                    logger.warning(f"🔁 Transformation search failed for '{alt_title}': {e}")
                    continue
                shortcut = exact_match_shortcut(alt_results, temp_search)
                if shortcut:
                    return shortcut, "exact_transform"
                orig_match = match_by_original_title(alt_results, temp_search, media_type)
                if orig_match:
                    return orig_match, "original_transform"
                alt = alternate_titles_fallback(alt_results, temp_search, media_type)
                if alt:
                    return alt, "alternate_transform"
                fuzzy = fuzzy_match_candidates(alt_results, temp_search, strict=True)
                if fuzzy:
                    return fuzzy, "fuzzy_transform"
                alt_candidates = []
                for res in alt_results:
                    alt_list = getattr(res, "alternative_titles", [])
                    for alt in alt_list:
                        alt_title2 = alt.get("title") if isinstance(alt, dict) else str(alt)
                        alt_candidates.append(SimpleNamespace(title=alt_title2, _orig_res=res))
                if alt_candidates:
                    fuzzy_alt_result = fuzzy_match_candidates(alt_candidates, temp_search, strict=True)
                    if fuzzy_alt_result:
                        fuzzy_alt = getattr(fuzzy_alt_result, "_orig_res", None)
                        if fuzzy_alt:
                            return fuzzy_alt, "fuzzy_alternate_transform"
    return None, None


@sleep_and_notify
@limits(calls=38, period=10)
def query_tmdb(search: MediaItem, media_type: str, retry: bool = False, retry_unidecode: bool = False, tried=None) -> Optional[Any]:
    if tried is None:
        tried = set()
    key = (search.title, search.year, media_type)
    is_new_attempt = key not in tried
    if not is_new_attempt:
        return None
    tried.add(key)
    _log_search_attempt(search.title, search.year, media_type, is_new_attempt)

    try:
        # === Step 1: Direct TMDB ID Lookup ===
        if getattr(search, "tmdb_id", None):
            result, reason = _try_id_lookup(search, media_type)
            if result:
                _log_result(
                    f"🎯 TMDB ID {reason} match:",
                    f"  → {getattr(result, 'title', getattr(result, 'name', ''))} ({search.year}) [{getattr(result, 'id', None)}] [{media_type}]",
                )
                search.match_reason = reason
                return result

        # === Step 2: Main TMDB Search ===
        search_results = perform_tmdb_search(search, media_type) or []
        if LOG_LEVEL == "DEBUG" and search_results:
            console(f"[DEBUG] Raw search results for “{search.title}” [{media_type}]:", "BLUE")
            for idx, res in enumerate(search_results, start=1):
                title = getattr(
                    res,
                    "title",
                    getattr(res, "name", getattr(res, "original_title", "")),
                )
                date = getattr(res, "release_date", getattr(res, "first_air_date", None))
                year_val = (
                    date.year if hasattr(date, "year") else (date[:4] if isinstance(date, str) and len(date) >= 4 else "None")
                )
                line = f"  {idx}. id={getattr(res,'id',None)}, title=\"{title}\", year={year_val}"
                console(line, "WHITE")
                logger.debug(line)

        result, reason = _try_main_match(search_results, search, media_type)
        if reason == "ambiguous":
            msg = f"🤷 Ambiguous result for “{search.title}” ({search.year}) [{media_type}] — Multiple possible matches found."
            logger.warning(msg)
            console(msg, "YELLOW")
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
            _log_result(
                f"🎯 {reason} match:",
                f"  → {getattr(result, 'title', getattr(result, 'name', ''))} ({year_val}) [{getattr(result, 'id', None)}] [{media_type}]",
            )
            search.match_reason = reason
            return result

        # === Step 4: Transformations ===
        if not retry and search_results:
            result, reason = _try_transformations(search_results, search, media_type)
            if result:
                year_val = search.year
                if hasattr(result, "release_date") or hasattr(result, "first_air_date"):
                    date_val = getattr(result, "release_date", getattr(result, "first_air_date", None))
                    if isinstance(date_val, str) and date_val[:4].isdigit():
                        year_val = int(date_val[:4])
                    elif hasattr(date_val, "year"):
                        year_val = date_val.year
                _log_result(
                    f"🎯 {reason} match:",
                    f"  → {getattr(result, 'title', getattr(result, 'name', ''))} ({year_val}) [{getattr(result, 'id', None)}] [{media_type}]",
                )
                search.match_reason = reason
                return result

        # === Step 3: Fallbacks ===
        if media_type == "movie":
            if is_new_attempt:
                logger.info(f"🔄 No confident match as movie; retrying as TV series for “{search.title}”")
                console(f"🔄 Retrying as TV series: “{search.title}”", "YELLOW")
            tv_result = query_tmdb(search, "tv_series", retry=retry, retry_unidecode=retry_unidecode, tried=tried)
            if tv_result:
                search.type = "tv_series"
                return tv_result

        msg = f"🤷 No confident match found for “{search.title}” ({search.year})"
        logger.warning(msg)
        console(msg)
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
        console(
            f"[WARNING] Failed to query TMDB for '{search.title}' ({search.year}) as {media_type}: {e}",
            "YELLOW",
        )
        logger.warning(f"Failed to query TMDB for '{search.title}' ({search.year}) as {media_type}: {e}")

        if "No Results Found" in str(e):
            result, reason = _try_transformations([], search, media_type)
            if result:
                _log_result(
                    f"🎯 {reason} match:",
                    f"  → {getattr(result, 'title', getattr(result, 'name', ''))} ({search.year}) [{getattr(result, 'id', None)}] [{media_type}]",
                )
                search.match_reason = reason
                return result

            if media_type == "movie" and hasattr(search, "files") and len(search.files) == 1:
                logger.info(f"🔄 Movie lookup failed; retrying as TV series for single-file “{search.title}”")
                console(f"🔄 Retrying as TV series: “{search.title}”", "YELLOW")
                tv_result = query_tmdb(search, "tv_series", tried=tried)
                if tv_result:
                    search.type = "tv_series"
                    return tv_result

            if media_type in ("movie", "tv_series") and not retry:
                logger.warning(f"🔁 Final fallback: retrying TMDB search with no year for “{search.title}”")
                console(f"🔁 Final fallback: retrying TMDB search with no year", "YELLOW")
                search.year = None
                return query_tmdb(search, media_type, retry=True, retry_unidecode=retry_unidecode, tried=tried)


def handle_data(config, items: list[MediaItem]) -> list[MediaItem]:
    """
    Enrich a list of MediaItem objects with metadata from TMDB.
    Updates UNMATCHED_CASES and TVDB_MISSING_CASES as needed.
    Args:
        items: List of MediaItem instances.
    Returns:
        The list of MediaItem instances (enriched).
    """
    print("🔄 Starting metadata enrichment via TMDB")
    logger.info("🔄 Starting metadata enrichment via TMDB")
    progress = None
    if config.quiet:
        progress = tqdm(total=len(items), desc="🔍 Enriching metadata", unit="item")

    # Load pending matches
    pending_matches = getattr(config, "pending_matches", {}).copy()

    for item in items:
        if getattr(config, "skip_collections", False) and getattr(item, "type", None) == "collection":
            logger.info(f"⏭️  Skipped collection: {item.title}")
            continue
        enriched = item.enrich()
        if config.quiet and progress is not None:
            progress.update(1)
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
            # Add to pending_matches if not present
            title_key = f"{item.title} ({item.year})" if item.year else item.title
            if title_key not in pending_matches:
                pending_matches[title_key] = "add_tmdb_url_here"
            continue
        # After enrichment, if item has a TMDB ID and is in pending_matches, remove it
        title_key = f"{item.title} ({item.year})" if item.year else item.title
        if getattr(item, "tmdb_id", None) and title_key in pending_matches:
            del pending_matches[title_key]
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
    # Save pending matches at the end
    save_pending_matches(pending_matches)

    if config.quiet and config.show_unmatched and UNMATCHED_CASES:
        for case in UNMATCHED_CASES:
            title = case.get("title", "Unknown")
            console(f"❌ Unmatched: {title}", "YELLOW")
    if config.quiet and progress is not None:
        progress.close()
    print("✅ Completed metadata enrichment")
    logger.info("✅ Completed metadata enrichment")
    return items


def sanitize_filename(name: str) -> str:
    """
    Sanitize a filename to be safe for most filesystems.
    Removes or replaces unsafe characters.
    Args:
        name: The filename string to sanitize.
    Returns:
        Sanitized filename string.
    """

    normalized = unicodedata.normalize("NFKD", name).encode("ASCII", "ignore").decode()

    cleaned = re.sub(r"[<>:\"/\\|?*]", "", normalized)

    allowed = set(string.ascii_letters + string.digits + r" !#$%&'()+,-.;=@[]^_`{}~")
    sanitized = "".join(c for c in cleaned if c in allowed)

    return sanitized.rstrip(" .")


def generate_new_filename(media_item: "MediaItem", old_filename: str) -> str:
    """
    Given a MediaItem and its old filename (basename), return the new sanitized filename
    with updated title, year, IDs, and season suffix.
    Args:
        media_item: The MediaItem instance.
        old_filename: The original filename (basename).
    Returns:
        The new sanitized filename as a string.
    """

    old_name_no_ext = os.path.splitext(old_filename)[0]
    base_title = media_item.new_title if media_item.new_title else (media_item.title if media_item.title else old_name_no_ext)
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

    new_name = sanitize_filename(new_name)
    new_name = " ".join(new_name.split()).strip()
    return new_name


def rename_files(items: list[MediaItem], config) -> tuple:
    """
    Rename files for all enriched MediaItem objects, respecting DRY_RUN mode.
    Handles filename conflicts, length limits, and logs all actions.
    In case of a conflict, keeps the file with the newest creation date,
    moves the older file to a global `duplicates` folder next to idarr.py, and logs this in a CSV.
    Args:
        items: List of MediaItem instances to process.
    Returns:
        Tuple: (List of tuples (media_type, old_filename, new_filename) for updated files, duplicate log list of dicts)
    """

    mode = "DRY RUN" if config.dry_run else "LIVE"
    logger.info(f"🏷  Starting file rename process ({mode} mode)")
    file_updates = []
    duplicate_log = []
    renamed, skipped = 0, 0

    duplicates_dir = os.path.join(SCRIPT_DIR, "duplicates")
    os.makedirs(duplicates_dir, exist_ok=True)
    if file_updates or LOG_LEVEL == "DEBUG":
        console(f"📂 Renaming files:")
        logger.info(f"📂 Renaming files:")
    for media_item in items:
        media_type = media_item.type
        if getattr(media_item, "match_failed", False):
            continue
        header_printed = False
        for index, file_path in enumerate(media_item.files):
            directory, old_filename = os.path.split(file_path)
            new_filename = generate_new_filename(media_item, old_filename)
            new_path = os.path.join(directory, new_filename)
            if old_filename == new_filename:
                if LOG_LEVEL == "DEBUG":
                    console(f"⏭️ Skipping unchanged file: {old_filename}", "YELLOW")
                    logger.debug(f"Skipping unchanged file: {old_filename}")
                skipped += 1
                continue
            if len(new_filename) > 255:
                console(f"⛔ Skipped (too long): {new_filename}", "RED")
                logger.warning(f"⛔ Skipped (too long): {new_filename}")
                skipped += 1
                continue
            conflict_path = os.path.join(directory, new_filename)
            if os.path.exists(conflict_path) and old_filename.lower() != new_filename.lower():
                src_stat = os.stat(file_path)
                dst_stat = os.stat(conflict_path)
                src_ctime = getattr(src_stat, "st_ctime", src_stat.st_mtime)
                dst_ctime = getattr(dst_stat, "st_ctime", dst_stat.st_mtime)
                if src_ctime >= dst_ctime:
                    keep_path, keep_file = file_path, old_filename
                    move_path, move_file = conflict_path, new_filename
                    action = "kept_source"
                else:
                    keep_path, keep_file = conflict_path, new_filename
                    move_path, move_file = file_path, old_filename
                    action = "kept_existing"
                base_move_file = os.path.basename(move_file)
                duplicate_dest = os.path.join(duplicates_dir, base_move_file)
                if os.path.exists(duplicate_dest):
                    ts = int(time.time())
                    name, ext = os.path.splitext(base_move_file)
                    duplicate_dest = os.path.join(duplicates_dir, f"{name}_{ts}{ext}")
                if not config.dry_run:
                    try:
                        shutil.move(move_path, duplicate_dest)
                        logger.info(f"🗂️ Conflict: Moved older file '{move_file}' to '{duplicate_dest}'")
                        console(
                            f"🗂️ Conflict: Moved older file '{move_file}' to '{duplicate_dest}'",
                            "YELLOW",
                        )
                    except Exception as e:
                        logger.error(f"❌ Failed to move duplicate '{move_file}': {e}")
                        skipped += 1
                        continue
                else:
                    logger.info(f"[DRY RUN] Would move older file '{move_file}' to '{duplicate_dest}'")
                    console(
                        f"[DRY RUN] Would move older file '{move_file}' to '{duplicate_dest}'",
                        "YELLOW",
                    )
                if src_ctime >= dst_ctime:
                    if not config.dry_run:
                        os.rename(file_path, conflict_path)
                    renamed += 1
                    file_updates.append((media_type, old_filename, new_filename))
                else:
                    skipped += 1
                duplicate_log.append(
                    {
                        "action": action,
                        "kept_file": os.path.basename(keep_file),
                        "kept_path": os.path.join(directory, keep_file),
                        "kept_ctime": (src_ctime if src_ctime >= dst_ctime else dst_ctime),
                        "moved_file": os.path.basename(move_file),
                        "moved_path": duplicate_dest,
                        "moved_ctime": (dst_ctime if src_ctime >= dst_ctime else src_ctime),
                    }
                )
                continue
            if config.dry_run:
                if media_type == "tv_series":
                    base_title = media_item.new_title if media_item.new_title is not None else media_item.title
                    base_year = media_item.new_year if media_item.new_year is not None else media_item.year
                    if not header_printed:
                        console(f"[DRY RUN] Series: {base_title} ({base_year})")
                        logger.info(f"[DRY RUN] Series: {base_title} ({base_year})")
                        header_printed = True
                    console(f"  • {old_filename} -> {new_filename}")
                    logger.info(f"  • {old_filename} -> {new_filename}")
                    if old_filename != new_filename:
                        console(f"\t- {old_filename}", "RED")
                        console(f"\t+ {new_filename}", "GREEN")
                        logger.info(f"\t- {old_filename}")
                        logger.info(f"\t+ {new_filename}")
                    renamed += 1
                    file_updates.append((media_type, old_filename, new_filename))
                    continue
                else:
                    msg = f"[DRY RUN] → {old_filename} -> {new_filename}"
                    console(msg)
                    logger.info(msg)
                    if old_filename != new_filename:
                        console(f"\t- {old_filename}", "RED")
                        console(f"\t+ {new_filename}", "GREEN")
                        logger.info(f"\t- {old_filename}")
                        logger.info(f"\t+ {new_filename}")
                    renamed += 1
                    file_updates.append((media_type, old_filename, new_filename))
                    continue
            else:
                try:
                    os.rename(file_path, new_path)
                    console(f"✅ Renamed: {old_filename} → {new_filename}")
                    logger.info(f"✅ Renamed: {old_filename} → {new_filename}")
                    renamed += 1
                    file_updates.append((media_type, old_filename, new_filename))
                except Exception as e:
                    console(f"❌ Failed to rename {old_filename}: {e}", "RED")
                    logger.error(f"❌ Failed to rename {old_filename}: {e}")
                    skipped += 1
    if config.dry_run:
        logger.info("")
        logger.info("📋 Rename Summary:")
        logger.info(f"  ✔️  {renamed} file(s) would be renamed")
        logger.info(f"  ⚠️  {skipped} file(s) skipped (conflicts or errors)")
        logger.info("")
    else:
        logger.info("")
        logger.info("📋 Rename Summary:")
        logger.info(f"  ✔️  {renamed} file(s) renamed")
        logger.info(f"  ⚠️  {skipped} file(s) skipped (conflicts or errors)")
        logger.info("")
    if file_updates:
        backup_path = os.path.join(LOG_DIR, "renamed_backup.json")
        with open(backup_path, "w") as f:
            json.dump(
                [{"old": old, "new": new, "type": typ} for typ, old, new in file_updates],
                f,
                indent=2,
            )
        logger.info(f"📝 Backup of renamed files written to {backup_path}")

    return file_updates, duplicate_log


def parse_args():
    parser = argparse.ArgumentParser(
        description="Enrich and rename media image files using TMDB metadata.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"idarr.py {FULL_VERSION}")

    general = parser.add_argument_group("General Options")
    general.add_argument(
        "--source",
        metavar="DIR",
        type=str,
        default=os.environ.get("SOURCE_DIR"),
        help="Directory of input image files",
    )
    general.add_argument(
        "--tmdb-api-key",
        metavar="KEY",
        type=str,
        default=os.environ.get("TMDB_API_KEY"),
        help="Override the TMDB API key",
    )
    general.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate renaming operations without making changes",
    )
    general.add_argument("--quiet", action="store_true", help="Suppress all output except progress bars")
    general.add_argument("--debug", action="store_true", help="Enable debug logging output")
    general.add_argument("--limit", metavar="N", type=int, help="Maximum number of items to process")
    general.add_argument(
        "--remove-non-image-files",
        action="store_true",
        help="If set, actually remove non-image files (default: ignore them)",
    )

    cache = parser.add_argument_group("Caching Options")
    cache.add_argument(
        "--frequency-days",
        metavar="DAYS",
        type=int,
        default=int(os.environ.get("FREQUENCY_DAYS", "30")),
        help="Days before cache entries are considered stale",
    )
    cache.add_argument(
        "--clear-cache",
        action="store_true",
        help="Delete the existing metadata cache before running",
    )
    cache.add_argument(
        "--cache-path",
        metavar="PATH",
        type=str,
        default=os.environ.get("CACHE_PATH"),
        help="Specify a custom cache file path",
    )
    cache.add_argument("--no-cache", action="store_true", help="Skip loading or saving the cache")

    filtering = parser.add_argument_group("Filtering Options")
    filtering.add_argument(
        "--filter",
        action="store_true",
        help="Enable filtering mode (requires one or more of --type, --year, or --contains)",
    )
    filtering.add_argument(
        "--type",
        choices=["movie", "tv_series", "collection"],
        help="Only process a specific media type",
    )
    filtering.add_argument(
        "--year",
        metavar="YEAR",
        type=int,
        help="Only process items released in a specific year",
    )
    filtering.add_argument(
        "--contains",
        metavar="TEXT",
        type=str,
        help="Only include titles containing this substring (case-insensitive)",
    )
    filtering.add_argument(
        "--id",
        metavar="ID",
        type=str,
        help="Only include items with a specific ID (tmdb-123, tvdb-456, imdb-tt1234567)",
    )
    filtering.add_argument(
        '--skip-collections',
        action='store_true',
        help='Skip enriching items identified as collections'
    )

    extra = parser.add_argument_group("Export & Recovery")
    extra.add_argument(
        "--show-unmatched",
        action="store_true",
        help="Print unmatched items even in quiet mode",
    )
    extra.add_argument(
        "--revert",
        action="store_true",
        help="Undo renames using the backup file (renamed_backup.json)",
    )
    args = parser.parse_args()
    return args


def load_runtime_config(args) -> SimpleNamespace:
    global TMDB_CLIENT, LOG_LEVEL
    config = SimpleNamespace()
    config.dry_run = getattr(args, "dry_run", False)
    config.quiet = getattr(args, "quiet", False)
    config.source_dir = args.source or os.environ.get("SOURCE_DIR")
    config.tmdb_api_key = args.tmdb_api_key or os.environ.get("TMDB_API_KEY")
    LOG_LEVEL = "DEBUG" if getattr(args, "debug", False) else "INFO"
    config.frequency_days = getattr(args, "frequency_days", None)
    if config.frequency_days is None:
        config.frequency_days = int(os.environ.get("FREQUENCY_DAYS", "30"))
    config.cache_path = (
        getattr(args, "cache_path", None) or os.environ.get("CACHE_PATH") or os.path.join(SCRIPT_DIR, "cache", "idarr_cache.json")
    )
    config.no_cache = getattr(args, "no_cache", False)
    config.clear_cache = getattr(args, "clear_cache", False)
    config.remove_non_image_files = getattr(args, "remove_non_image_files", False)
    config.limit = getattr(args, "limit", None)
    config.filter = getattr(args, "filter", False)
    config.type = getattr(args, "type", None)
    config.year = getattr(args, "year", None)
    config.contains = getattr(args, "contains", None)
    config.id = getattr(args, "id", None)
    config.show_unmatched = getattr(args, "show_unmatched", False)
    config.revert = getattr(args, "revert", False)
    config.skip_collections = getattr(args, "skip_collections", False)

    if config.no_cache:
        config.cache = {}
    else:
        config.cache = load_cache()
    if config.clear_cache:
        if os.path.exists(config.cache_path):
            os.remove(config.cache_path)
        config.cache = {}
    # Load pending matches
    config.pending_matches = load_pending_matches()
    if not config.tmdb_api_key:
        raise RuntimeError("TMDB API key is required. Set via --tmdb-api-key or TMDB_API_KEY environment variable.")
    TMDB_CLIENT = TMDbAPIs(config.tmdb_api_key)

    return config


def print_settings(config):
    console = Console()

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Section", style="cyan", no_wrap=True)
    table.add_column("Setting", style="green")
    table.add_column("Value", style="white")

    table.add_row("General", "SOURCE_DIR", str(config.source_dir))
    table.add_row("General", "DRY_RUN", str(getattr(config, "dry_run", False)))
    table.add_row("General", "QUIET", str(getattr(config, "quiet", False)))
    table.add_row("General", "LOG_LEVEL", str(getattr(config, "log_level", "INFO")))
    table.add_row("General", "LIMIT", str(getattr(config, "limit", None)))

    table.add_row("Caching", "CACHE_PATH", str(getattr(config, "cache_path", "")))
    table.add_row("Caching", "NO_CACHE", str(getattr(config, "no_cache", False)))
    table.add_row("Caching", "CLEAR_CACHE", str(getattr(config, "clear_cache", False)))
    table.add_row("Caching", "FREQUENCY_DAYS", str(getattr(config, "frequency_days", 30)))

    table.add_row(
        "Export/Recovery",
        "SHOW_UNMATCHED",
        str(getattr(config, "show_unmatched", False)),
    )
    table.add_row("Export/Recovery", "REVERT", str(getattr(config, "revert", False)))

    table.add_row(
        "TMDB",
        "TMDB_API_KEY",
        "********" if getattr(config, "tmdb_api_key", None) else None,
    )

    if getattr(config, "filter", False):
        table.add_row("Filtering", "FILTER", "True")
        if getattr(config, "type", None) is not None:
            table.add_row("Filtering", "TYPE", str(config.type))
        if getattr(config, "year", None) is not None:
            table.add_row("Filtering", "YEAR", str(config.year))
        if getattr(config, "contains", None) is not None:
            table.add_row("Filtering", "CONTAINS", str(config.contains))
        if getattr(config, "id", None) is not None:
            table.add_row("Filtering", "ID", str(config.id))

    console.print("[bold blue]🔧 Current Settings[/bold blue]\n")
    console.print(table)


def perform_revert(config, items: list[MediaItem]) -> bool:
    if config.revert:
        revert_path = os.path.join(LOG_DIR, "renamed_backup.json")
        if not os.path.exists(revert_path):
            return True
        with open(revert_path) as f:
            entries = json.load(f)
        filtered_filenames = {os.path.basename(fp) for item in items for fp in item.files}
        for entry in entries:
            if entry["new"] not in filtered_filenames:
                continue
            old_path = os.path.join(config.source_dir, entry["old"])
            new_path = os.path.join(config.source_dir, entry["new"])
            if os.path.exists(new_path):
                try:
                    os.rename(new_path, old_path)
                    console(f"↩️ Reverted: {entry['new']} → {entry['old']}", "YELLOW")
                except Exception as e:
                    logger.error(f"Failed to revert {entry['new']}: {e}")
        return True
    return False


def filter_items(args, items: list[MediaItem]) -> list[MediaItem]:
    if args.filter:
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
                console(
                    f"❌ Invalid --id format. Use tmdb-123, tvdb-456, or imdb-tt1234567",
                    "RED",
                )
                exit(1)
    if args.limit:
        items = items[: args.limit]
    return items


def export_csvs(updated_items: list, file_updates: list, duplicate_log: list) -> None:
    """Export all CSVs (updated_files, unmatched_cases, duplicates_log) with shared logic."""

    def write_csv(path, rows, fieldnames):
        if not rows:
            return
        with open(path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        logger.info(f"CSV written to {path}")
        console(f"⚙️ CSV written to {path}")

    # Updated Files CSV
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

    # Unmatched Cases CSV
    unmatched_cases_path = os.path.join(LOG_DIR, "unmatched_cases.csv")
    unmatched_fieldnames = ["files", "title", "year", "media_type", "tmdb_id", "imdb_id", "tvdb_id", "match_reason"]
    # Dynamically extend fields if there are extras
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

    # Duplicate Log CSV
    duplicates_csv_path = os.path.join(LOG_DIR, "duplicates_log.csv")
    duplicate_fieldnames = ["action", "kept_file", "kept_path", "kept_ctime", "moved_file", "moved_path", "moved_ctime"]
    # Add any extras found in duplicate_log entries
    all_dupe_keys = set()
    for entry in duplicate_log:
        all_dupe_keys.update(entry.keys())
    extra_dupe = [k for k in sorted(all_dupe_keys) if k not in duplicate_fieldnames]
    final_duplicate_fieldnames = duplicate_fieldnames + extra_dupe

    write_csv(duplicates_csv_path, duplicate_log, final_duplicate_fieldnames)

    # TVDB Missing Cases CSV
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


def summarize_run(
    start_time: float,
    items: list[MediaItem],
    updated_items: list[MediaItem],
    file_updates: list,
    config,
) -> None:
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

    console_rich = Console()
    table = Table(show_header=False, box=None, padding=(0, 1))
    for label, value in labels:
        table.add_row(label, str(value))
    console_rich.rule("[bold]Summary Report")
    console_rich.print(table)
    console_rich.rule()
    logger.info("Summary Report:")
    for label, value in labels:
        logger.info(f"{label}: {value}")
    active_keys = {get_cache_key(item) for item in updated_items}
    if not config.no_cache:
        save_cache(config.cache, active_keys)


def main():
    args = parse_args()
    if getattr(args, "filter", False):
        if not (args.type or args.year or args.contains):
            console(
                "❌ --filter requires at least one of --type, --year, or --contains",
                "RED",
            )
            exit(1)
    config = load_runtime_config(args)
    logger.info(f"************************* IDARR Version: {FULL_VERSION} *************************")
    if LOG_LEVEL == "DEBUG":
        print_settings(config)
    if getattr(args, "revert", False):
        items = scan_files_in_flat_folder(config)
        items = filter_items(args, items)
        perform_revert(config, items)
        return
    start_time = time.time()
    items = scan_files_in_flat_folder(config)
    items = filter_items(args, items)
    updated_items = handle_data(config, items)
    file_updates, duplicate_log = rename_files(updated_items, config)
    export_csvs(updated_items, file_updates, duplicate_log)
    summarize_run(start_time, items, updated_items, file_updates, config)


if __name__ == "__main__":
    main()
