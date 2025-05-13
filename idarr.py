# Version 1.0.0
### === CONFIGURATION === ###
DRY_RUN = True
QUIET = False  # Suppress console output and use progress bar instead
source_dir = "/Users/drazzilb/sandbox/input/test_folder"
# You can set YOUR_TMDB_API_KEY here, or via the environment variable "TMDB_API_KEY"
YOUR_TMDB_API_KEY = ""
LOG_LEVEL = "INFO"


FREQUENCY_DAYS = 30

### === Do not edit below here === ###


import sys

# Require Python 3.8+
if sys.version_info < (3, 10):
    print("Python 3.10 or higher is required. Detected version: {}.{}.{}".format(*sys.version_info[:3]))
    exit(1)
try:
    from tmdbapis import TMDbAPIs
    from ratelimit import limits, RateLimitException
    from tqdm import tqdm
    from unidecode import unidecode
    from dotenv import load_dotenv
except ImportError:
    print("requirements are not installed. Please install all dependencies with 'pip install -r requirements.txt'.")
    exit(1)

import os
import re
import csv
import logging
import unicodedata
import string
import time
from typing import List, Dict, Pattern, Optional, Tuple, Any
from pprint import pprint
from collections import defaultdict
from difflib import SequenceMatcher
from functools import wraps
import json
from datetime import datetime, timedelta

from concurrent.futures import ThreadPoolExecutor

# === Cache utilities ===
CACHE_PATH = os.path.expanduser("~/.cache/idarr_cache.json")

def load_cache() -> Dict[str, Any]:
    """
    Load the cache from the cache file if it exists.
    Returns:
        A dictionary representing the cache.
    """
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, 'r') as f:
            return json.load(f)
    return {}

def save_cache(cache: Dict[str, Any], active_keys: set) -> None:
    """
    Save the cache to the cache file, filtering to only include active keys.
    Args:
        cache: The current cache dictionary.
        active_keys: Set of keys to retain in the cache.
    """
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    cache = {k: v for k, v in cache.items() if k in active_keys}
    with open(CACHE_PATH, 'w') as f:
        json.dump(cache, f, indent=2)

def is_recent(last_checked: str) -> bool:
    """
    Determine if a cached entry is recent, based on FREQUENCY_DAYS.
    Args:
        last_checked: ISO format date string.
    Returns:
        True if last_checked is within FREQUENCY_DAYS, else False.
    """
    try:
        last_time = datetime.fromisoformat(last_checked)
        return datetime.now() - last_time < timedelta(days=FREQUENCY_DAYS)
    except Exception:
        return False

# === Console Helper ===
def console(msg: str, level: str = "WHITE") -> None:
    """
    Print a message to the console with optional color, if not in QUIET mode.
    Args:
        msg: Message string to print.
        level: Color level as a string.
    """
    colors = {
        "WHITE": "\033[97m",      # White
        "YELLOW": "\033[93m",     # yellow
        "RED": "\033[91m",        # red
        "BLUE": "\033[94m",       # blue
        "GREEN": "\033[92m",      # Green
    }
    end = "\033[0m"
    if not QUIET:
        print(f"{colors.get(level,'')}{msg}{end}")

# === NoResultsError for TMDB lookups ===
class NoResultsError(Exception):
    """Custom exception for TMDB lookups that return no results."""
    pass

# === MediaItem class ===
class MediaItem:
    """
    Represents a media item (movie, tv_series, or collection) with associated metadata and files.
    """
    def __init__(
        self,
        type: str,
        title: str,
        year: Optional[int],
        tmdb_id: Optional[int],
        tvdb_id: Optional[int] = None,
        imdb_id: Optional[str] = None,
        files: Optional[List[str]] = None
    ) -> None:
        self.type = type
        self.title = title
        self.year = year
        self.tmdb_id = tmdb_id
        self.tvdb_id = tvdb_id
        self.imdb_id = imdb_id
        self.files = files or []
        # Placeholders for updated metadata after enrichment
        self.new_title: Optional[str] = None
        self.new_year: Optional[int] = None
        self.new_tmdb_id: Optional[int] = None
        self.new_tvdb_id: Optional[int] = None
        self.new_imdb_id: Optional[str] = None
        self.match_failed: bool = False
        self.match_reason: Optional[str] = None

    def enrich(self) -> bool:
        """
        Perform TMDB lookup and update self.* fields with enriched metadata.
        Uses cache if available and recent.
        Returns:
            True if enrichment succeeded, False otherwise.
        """
        cache_key = f"{self.title} ({self.year}) [{self.type}]"
        cached = CACHE.get(cache_key)
        should_skip = not DRY_RUN and cached and is_recent(cached.get("last_checked", ""))

        if not QUIET:
            console("")
        logger.info("")
        # --- cache check ---
        if should_skip:
            if cached.get("no_result"):
                logger.info(f"📦 Skipping (previously not found): {cache_key}")
                return False
            self.new_title = cached.get("title")
            self.new_year = cached.get("year")
            self.new_tmdb_id = cached.get("tmdb_id")
            self.new_tvdb_id = cached.get("tvdb_id")
            self.new_imdb_id = cached.get("imdb_id")
            logger.info(f"📦 Used cached metadata for {cache_key}")
            return True
        result = query_tmdb(self, self.type)
        if not result:
            # cache "no result" so we skip in future runs
            CACHE[cache_key] = {
                "last_checked": datetime.now().isoformat(),
                "no_result": True
            }
            self.match_failed = True
            return False
        # update IDs if they differ from current
        if hasattr(result, 'id') and result.id != self.tmdb_id:
            if result.id and result.id != self.tmdb_id:
                if not QUIET:
                    console(f"⚠️ TMDB ID mismatch: {self.tmdb_id} → {result.id}", "YELLOW")
                logger.warning(f"⚠️ TMDB ID mismatch: {self.tmdb_id} → {result.id}")
            self.new_tmdb_id = result.id
        if hasattr(result, 'tvdb_id') and result.tvdb_id != self.tvdb_id:
            if getattr(result, 'tvdb_id', None) and result.tvdb_id != self.tvdb_id:
                if not QUIET:
                    console(f"⚠️ TVDB ID mismatch: {self.tvdb_id} → {result.tvdb_id}", "YELLOW")
                logger.warning(f"⚠️ TVDB ID mismatch: {self.tvdb_id} → {result.tvdb_id}")
            self.new_tvdb_id = result.tvdb_id
        if hasattr(result, 'imdb_id') and result.imdb_id != self.imdb_id:
            if getattr(result, 'imdb_id', None) and result.imdb_id != self.imdb_id:
                if not QUIET:
                    console(f"⚠️ IMDB ID mismatch: {self.imdb_id} → {result.imdb_id}", "YELLOW")
                logger.warning(f"⚠️ IMDB ID mismatch: {self.imdb_id} → {result.imdb_id}")
            self.new_imdb_id = result.imdb_id
        # update title/year if different
        tmdb_title = getattr(result, 'title', getattr(result, 'name', None))
        tmdb_date = getattr(result, 'first_air_date', getattr(result, 'release_date', None))
        if tmdb_date:
            res_year = tmdb_date.year if hasattr(tmdb_date, 'year') else int(str(tmdb_date)[:4])
            if res_year != self.year:
                self.new_year = res_year
        if tmdb_title and tmdb_title != self.title:
            self.new_title = tmdb_title
        # --- update cache ---
        CACHE[cache_key] = {
            "last_checked": datetime.now().isoformat(),
            "type": self.type,
            "tmdb_id": self.new_tmdb_id or self.tmdb_id,
            "tvdb_id": self.new_tvdb_id or self.tvdb_id,
            "imdb_id": self.new_imdb_id or self.imdb_id,
            "year": self.new_year or self.year,
            "title": self.new_title or self.title
        }
        # Show match result if not QUIET
        if not QUIET:
            match_strs = []
            if self.new_title and self.new_title != self.title:
                match_strs.append(f"title: {self.title} → {self.new_title}")
            if self.new_year and self.new_year != self.year:
                match_strs.append(f"year: {self.year} → {self.new_year}")
            if self.new_tmdb_id and self.new_tmdb_id != self.tmdb_id:
                match_strs.append(f"tmdb_id: {self.tmdb_id} → {self.new_tmdb_id}")
            if self.new_tvdb_id and self.new_tvdb_id != self.tvdb_id:
                match_strs.append(f"tvdb_id: {self.tvdb_id} → {self.new_tvdb_id}")
            if self.new_imdb_id and self.new_imdb_id != self.imdb_id:
                match_strs.append(f"imdb_id: {self.imdb_id} → {self.new_imdb_id}")
            if match_strs:
                console("✅ Metadata enriched: " + "; ".join(match_strs), "GREEN")
        return True

    def needs_rename(self) -> bool:
        """
        Determine if any new_* fields exist, indicating a rename is needed.
        Returns:
            True if rename is needed, False otherwise.
        """
        return any([self.new_title, self.new_year, self.new_tmdb_id, self.new_tvdb_id, self.new_imdb_id])

    def filenames(self) -> List[Tuple[str, str]]:
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

# === sleep_and_notify decorator for rate limit warning ===
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
                console(f"\033[93m[WARNING]\033[0m Rate limit hit, sleeping for {e.period_remaining:.2f} seconds", "RED")
                time.sleep(e.period_remaining)
    return wrapper

dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(dotenv_path)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
YEAR_REGEX: Pattern = re.compile(r"\s?\((\d{4})\)(?!.*Collection).*")
SEASON_PATTERN: Pattern = re.compile(
    r"(?:\s*-\s*Season\s*\d+|_Season\d{1,2}|\s*-\s*Specials|_Specials)", re.IGNORECASE
)
TMDB_ID_REGEX: Pattern = re.compile(r"tmdb[-_\s](\d+)")
TVDB_ID_REGEX: Pattern = re.compile(r"tvdb[-_\s](\d+)")
IMDB_ID_REGEX: Pattern = re.compile(r"imdb[-_\s](tt\d+)")

UNMATCHED_CASES: List[Dict[str, Any]] = []
TVDB_MISSING_CASES: List[Dict[str, Any]] = []
# Track movie→tv reclassifications
RECLASSIFIED: List[Dict[str, Any]] = []
TMDB_API_KEY = os.environ.get("TMDB_API_KEY", YOUR_TMDB_API_KEY)

# Create a single TMDb client for reuse
tmdb_client = TMDbAPIs(TMDB_API_KEY)

# DEBUG_MODE flag: True if LOG_LEVEL is DEBUG
DEBUG_MODE = LOG_LEVEL.upper() == "DEBUG"

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp"]

# === LOGGER SETUP ===
LOG_DIR = os.path.join(SCRIPT_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
log_path = os.path.join(LOG_DIR, "idarr.log")

# === CACHE PATH (for future use) ===
# Use script-relative cache directory, not ~/.cache
CACHE_PATH = os.path.join(SCRIPT_DIR, "cache", "idarr_cache.json")

logger = logging.getLogger(__name__)
logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.DEBUG))

log_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# === ANSI Color Filter for file logs ===
class RemoveColorFilter(logging.Filter):
    ansi_escape = re.compile(r'\x1B[@-_][0-?]*[ -/]*[@-~]')
    def filter(self, record):
        if isinstance(record.msg, str):
            record.msg = self.ansi_escape.sub('', record.msg)
        return True

file_handler = logging.FileHandler(log_path, mode='w')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(log_formatter)
file_handler.addFilter(RemoveColorFilter())
logger.addHandler(file_handler)


def sanitize_filename(name: str) -> str:
    """
    Sanitize a filename to be safe for most filesystems.
    Removes or replaces unsafe characters.
    Args:
        name: The filename string to sanitize.
    Returns:
        Sanitized filename string.
    """
    # Normalize to ASCII
    normalized = unicodedata.normalize('NFKD', name).encode('ASCII', 'ignore').decode()
    # Remove Windows-reserved chars: < > : " / \ | ? *
    cleaned = re.sub(r'[<>:\"/\\|?*]', '', normalized)
    # Keep only safe characters: letters, digits, space, dash, underscore, dot, parentheses, curly braces
    allowed = set(string.ascii_letters + string.digits + " -_.(){}")
    sanitized = ''.join(c for c in cleaned if c in allowed)
    # Trim trailing spaces and dots
    return sanitized.rstrip(' .')

# === generate_new_filename helper ===
def generate_new_filename(media_item: 'MediaItem', old_filename: str) -> str:
    """
    Given a MediaItem and its old filename (basename), return the new sanitized filename
    with updated title, year, IDs, and season suffix.
    Args:
        media_item: The MediaItem instance.
        old_filename: The original filename (basename).
    Returns:
        The new sanitized filename as a string.
    """
    # Determine base title and year, falling back to old filename if necessary
    old_name_no_ext = os.path.splitext(old_filename)[0]
    base_title = (
        media_item.new_title
        if media_item.new_title
        else (media_item.title if media_item.title else old_name_no_ext)
    )
    base_year = (
        media_item.new_year
        if media_item.new_year is not None
        else media_item.year
    )
    # Build ID suffixes
    id_parts = []
    for attr, prefix in (('new_tmdb_id', 'tmdb'), ('new_tvdb_id', 'tvdb'), ('new_imdb_id', 'imdb')):
        val = getattr(media_item, attr, None) or getattr(media_item, attr.replace('new_', ''), None)
        if val:
            id_parts.append(f"{prefix}-{val}")
    suffix = ''.join(f" {{{part}}}" for part in id_parts)
    # Extract season suffix
    season_suffix = ''
    match = SEASON_PATTERN.search(old_filename)
    if match:
        season_suffix = match.group(0)
    # Split name and extension
    name, ext = os.path.splitext(old_filename)
    # Construct base
    base = f"{base_title}{f' ({base_year})' if base_year else ''}"
    # Place suffix relative to season for TV series
    if media_item.type == "tv_series":
        new_name = f"{base}{suffix}{season_suffix}{ext}"
    else:
        new_name = f"{base}{season_suffix}{suffix}{ext}"
    # Sanitize and collapse whitespace
    new_name = sanitize_filename(new_name)
    new_name = ' '.join(new_name.split()).strip()
    return new_name


# === Helper for accent-insensitive, punctuation-insensitive normalization ===
def normalize_str(s: str) -> str:
    """
    Normalize a string by removing diacritics, non-alphanumeric characters, and lowercasing.
    Args:
        s: Input string.
    Returns:
        Normalized string.
    """
    nfkd = unicodedata.normalize('NFKD', s)
    only_ascii = nfkd.encode('ASCII', 'ignore').decode()
    return re.sub(r'[^a-z0-9]', '', only_ascii.lower())


# === Normalization with common aliases/abbreviations ===
def normalize_with_aliases(s: str) -> str:
    """
    Normalize a string to ASCII, lowercased, with common abbreviations/aliases expanded.
    Tries both abbreviation→full and full→abbreviation substitutions.
    Returns a single canonical normalized form.
    Args:
        s: Input string.
    Returns:
        Normalized string with aliases handled.
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
    ]
    nfkd = unicodedata.normalize('NFKD', s)
    s_ascii = nfkd.encode('ASCII', 'ignore').decode()
    norm_set = set()
    for a, b in substitutions:
        # Try both abbreviation-to-full and full-to-abbreviation forms
        alt1 = re.sub(rf"\b{re.escape(a)}\b", b, s_ascii, flags=re.IGNORECASE)
        alt2 = re.sub(rf"\b{re.escape(b)}\b", a, s_ascii, flags=re.IGNORECASE)
        norm_set.add(re.sub(r'[^a-z0-9]', '', alt1.lower()))
        norm_set.add(re.sub(r'[^a-z0-9]', '', alt2.lower()))
    # Also include original normalized form
    norm_set.add(re.sub(r'[^a-z0-9]', '', s_ascii.lower()))
    # Return all forms joined so they can be compared for any-match
    return list(norm_set)[0]  # Primary normalized string

def create_collection(
    title: str,
    imdb_id: Optional[str],
    tmdb_id: Optional[int],
    files: List[str]
) -> Dict[str, Any]:
    """
    Create a collection dictionary for MediaItem initialization.
    """
    return {
        'type': 'collection',
        'title': title,
        'year': None,
        'imdb_id': imdb_id,
        'tmdb_id': tmdb_id,
        'files': files,
    }

def create_series(
    title: str,
    year: Optional[int],
    tvdb_id: Optional[int],
    imdb_id: Optional[str],
    tmdb_id: Optional[int],
    files: List[str]
) -> Dict[str, Any]:
    """
    Create a tv_series dictionary for MediaItem initialization.
    """
    return {
        'type': 'tv_series',
        'title': title,
        'year': year,
        'tvdb_id': tvdb_id,
        'imdb_id': imdb_id,
        'tmdb_id': tmdb_id,
        'files': files,
    }

def create_movie(
    title: str,
    year: Optional[int],
    tmdb_id: Optional[int],
    imdb_id: Optional[str],
    files: List[str]
) -> Dict[str, Any]:
    """
    Create a movie dictionary for MediaItem initialization.
    """
    return {
        'type': 'movie',
        'title': title,
        'year': year,
        'tmdb_id': tmdb_id,
        'imdb_id': imdb_id,
        'files': files,
    }

def extract_ids(text: str) -> Tuple[Optional[int], Optional[int], Optional[str]]:
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

def parse_file_group(folder_path: str, base_name: str, files: List[str]) -> MediaItem:
    """
    Parse a group of files and return a MediaItem instance.
    Args:
        folder_path: Directory path containing the files.
        base_name: Base name representing the group.
        files: List of filenames in the group.
    Returns:
        MediaItem instance.
    """
    # Strip known ID suffixes and year from title
    title = re.sub(r"{(tmdb|tvdb|imdb)-[^}]+}", "", base_name)
    title = re.sub(YEAR_REGEX, '', title).strip()
    year_match = YEAR_REGEX.search(base_name)
    year = int(year_match.group(1)) if year_match else None
    tmdb_id, tvdb_id, imdb_id = extract_ids(base_name)
    files = sorted([os.path.join(folder_path, file) for file in files if not file.startswith('.')])
    is_series = any(SEASON_PATTERN.search(file) for file in files)
    is_collection = not year
    if is_collection:
        data = create_collection(title, tmdb_id, imdb_id, files)
    elif is_series:
        data = create_series(title, year, tvdb_id, imdb_id, tmdb_id, files)
    else:
        data = create_movie(title, year, tmdb_id, imdb_id, files)
    return MediaItem(**data)


def scan_files_in_flat_folder(source_dir: str) -> List[MediaItem]:
    """
    Scan a flat folder for image assets and group them into MediaItem instances.
    Args:
        source_dir: Directory to scan.
    Returns:
        List of MediaItem instances.
    """
    logger.info(f"📂 Scanning directory for image assets: {source_dir}")
    try:
        files = os.listdir(source_dir)
    except FileNotFoundError:
        return []
    groups = defaultdict(list)
    assets_dict = []
    for file in files:
        if file.startswith('.'):
            continue
        ext = os.path.splitext(file)[-1].lower()
        if ext not in IMAGE_EXTENSIONS:
            full_path = os.path.join(source_dir, file)
            if DRY_RUN:
                logger.info(f"[DRY RUN] Would delete non-image file: {file}")
            else:
                try:
                    os.remove(full_path)
                    logger.info(f"🗑️ Removed non-image file: {file}")
                except Exception as e:
                    logger.error(f"❌ Failed to delete {file}: {e}")
            continue
        title = file.rsplit('.', 1)[0]
        raw_title = SEASON_PATTERN.split(title)[0].strip()
        groups[raw_title].append(file)
    groups = dict(sorted(groups.items(), key=lambda x: x[0].lower()))
    # Flatten all files for progress bar
    all_files = [file for group in groups.values() for file in group if not file.startswith('.')]
    with tqdm(total=len(all_files), desc=f'Processing files {os.path.basename(source_dir)}', unit='file') as progress:
        for base_name, files in groups.items():
            assets_dict.append(parse_file_group(source_dir, base_name, files))
            progress.update(len(files))
    total_assets = sum(len(v) for v in groups.values())
    logger.info(f"✅ Completed scanning: discovered {len(assets_dict)} asset groups covering {total_assets} files")
    # Return flat list of items as MediaItem
    return assets_dict

def exact_match_shortcut(search_results: List[Any], search: MediaItem) -> Optional[Any]:
    norm_search = normalize_with_aliases(search.title)
    for res in search_results:
        title = getattr(res,"title",getattr(res,"name",""))
        if normalize_with_aliases(title)==norm_search:
            date = getattr(res,"release_date",getattr(res,"first_air_date",""))
            year = date.year if hasattr(date,"year") else (int(date[:4]) if isinstance(date,str) and date[:4].isdigit() else None)
            if year==search.year:
                return res
    return None

def alternate_titles_fallback(search_results: List[Any], search: MediaItem, media_type: str) -> Optional[Any]:
    """
    Use pre-fetched alternate_titles from each search result, without new API calls.
    """
    norm_search = normalize_with_aliases(search.title)
    for res in search_results:
        # Attempt to get the preloaded alternate_titles list from the result
        alt_list = getattr(res, "alternative_titles", [])
        for alt in alt_list:
            # alt could be a dict with 'title' or a simple string
            cand = alt.get("title") if isinstance(alt, dict) else str(alt)
            if normalize_with_aliases(cand) == norm_search:
                return res
    return None

def high_confidence_fuzzy(search_results: List[Any], search: MediaItem) -> Optional[Any]:
    def word_jaccard(a: str, b: str) -> float:
        words_a = set(re.findall(r'\w+', a.lower()))
        words_b = set(re.findall(r'\w+', b.lower()))
        if not words_a or not words_b:
            return 0.0
        intersection = words_a & words_b
        union = words_a | words_b
        return len(intersection) / len(union)

    norm_search = normalize_with_aliases(search.title)
    candidates = []
    for res in search_results:
        title = getattr(res, 'title', getattr(res, 'name', ''))
        norm_res = normalize_with_aliases(title)
        # Fuzzy ratio
        ratio = SequenceMatcher(None, norm_search, norm_res).ratio()
        # Year extraction
        date = getattr(res, 'release_date', getattr(res, 'first_air_date', ''))
        res_year = None
        if isinstance(date, str) and date[:4].isdigit():
            res_year = int(date[:4])
        elif hasattr(date, "year"):
            res_year = date.year
        # Word-level similarity
        jaccard = word_jaccard(search.title, title)
        # Acceptance criteria: high ratio, exact year, strong word overlap (allowing a single extra word)
        if ratio >= 0.95 and res_year == search.year and jaccard >= 0.85:
            candidates.append(res)
    return candidates[0] if len(candidates) == 1 else None

def fuzzy_scoring(search_results: List[Any], search: MediaItem) -> Tuple[List[Any], List[Tuple[float, Any]]]:
    scored=[]
    norm_search=normalize_with_aliases(search.title)
    for res in search_results:
        title = getattr(res,"title",getattr(res,"name",""))
        norm_res = normalize_with_aliases(title)
        t_score = SequenceMatcher(None,norm_search,norm_res).ratio()
        date = getattr(res,"release_date",getattr(res,"first_air_date",""))
        year = date.year if hasattr(date,"year") else (int(date[:4]) if isinstance(date,str) and date[:4].isdigit() else None)
        y_score = 1.0 if year==search.year else 0.5 if year and search.year and abs(year-search.year)<=1 else 0
        scored.append((t_score*2 + y_score, res))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [res for _,res in scored if _>1.0], scored

def perform_tmdb_search(search: MediaItem, media_type: str) -> Optional[List[Any]]:
    tmdb = tmdb_client
    if media_type == "collection":
        return tmdb.collection_search(query=search.title)
    elif media_type == "movie":
        return tmdb.movie_search(query=search.title, year=search.year)
    elif media_type == "tv_series":
        return tmdb.tv_search(query=search.title, first_air_date_year=search.year)
    else:
        msg = f"[SKIPPED] Unsupported media type '{media_type}' for '{search.title}'"
        logger.info(msg)
        return None

def match_by_id(search_results: List[Any], search: MediaItem, media_type: str) -> Optional[Any]:
    for res in search_results:
        if (
            (getattr(search, "tmdb_id", None) and getattr(res, "id", None) == search.tmdb_id) or
            (getattr(search, "tvdb_id", None) and getattr(res, "tvdb_id", None) == search.tvdb_id) or
            (getattr(search, "imdb_id", None) and getattr(res, "imdb_id", None) == search.imdb_id)
        ):
            header = "🎯 ID-based match:"
            msg = f"  → {getattr(res, 'title', getattr(res, 'name', ''))} ({search.year}) [{getattr(res, 'id', None)}] [{media_type}]"
            logger.info(header)
            logger.info(msg)
            console(header)
            console(msg, "GREEN")
            return res
    return None

def match_by_original_title(search_results: List[Any], search: MediaItem, media_type: str) -> Optional[Any]:
    for res in search_results:
        orig_title = getattr(res, 'original_title', None)
        if orig_title and normalize_with_aliases(orig_title) == normalize_with_aliases(search.title):
            header = "🎯 Original-title:"
            msg = f"  → {orig_title} ({search.year}) [{getattr(res, 'id', None)}] [{media_type}]"
            logger.info(header)
            logger.info(msg)
            console(header)
            console(msg, "GREEN")
            return res
    return None

def attempt_fallbacks(search_results: List[Any], search: MediaItem, media_type: str) -> Optional[Any]:
    # Alternate titles
    alt = alternate_titles_fallback(search_results, search, media_type)
    if alt:
        title_str = getattr(alt, 'title', getattr(alt, 'name', ''))
        header = "🔄 Alternate-title match:"
        msg = f"  → {title_str} ({search.year}) [{alt.id}] [{media_type}]"
        logger.info(header)
        logger.info(msg)
        console(header)
        console(msg, "GREEN")
        return alt
    # High-confidence fuzzy
    high = high_confidence_fuzzy(search_results, search)
    if high:
        title_str = getattr(high, 'title', getattr(high, 'name', ''))
        header = "🎯 High-confidence fuzzy match:"
        msg = f"  → {title_str} ({search.year}) [{getattr(high, 'id', None)}] [{media_type}]"
        logger.info(header)
        logger.info(msg)
        console(header)
        console(msg, "GREEN")
        return high
    return None

@sleep_and_notify
@limits(calls=38, period=10)
def query_tmdb(search: dict, media_type: str, retry: bool = False, retry_unidecode: bool = False) -> Optional[Any]:
    """
    Query TMDB for a given media item, attempting to find the best match using a series of matching strategies.
    Steps are clearly marked for clarity and maintainability.
    """
    orig_title = search.title
    selected_id = None
    try:
        logger.info(f"🔍 Searching TMDB for “{search.title}” ({search.year}) [{media_type}]...")
        console(f"🔍 Searching TMDB for “{search.title}” ({search.year}) [{media_type}]...")

        # === Step 1: Search TMDB ===
        # Perform search by media type (movie, tv_series, collection, etc.)
        search_results = perform_tmdb_search(search, media_type)
        if search_results is None:
            return None

        # Debug output for search results (if enabled)
        if DEBUG_MODE and search_results:
            console(f"[DEBUG] Raw search results for “{orig_title}” [{media_type}]:", "BLUE")
            for idx, res in enumerate(search_results, start=1):
                title = getattr(res, 'title', getattr(res, 'name', getattr(res, 'original_title', '')))
                date = getattr(res, 'release_date', getattr(res, 'first_air_date', None))
                year_val = date.year if hasattr(date, 'year') else (date[:4] if isinstance(date, str) and len(date) >= 4 else 'None')
                line = f"  {idx}. id={getattr(res,'id',None)}, title=\"{title}\", year={year_val}"
                if getattr(res, 'id', None) == selected_id:
                    console(line, "GREEN")
                else:
                    console(line, "WHITE")

        # === Step 2: Handle No Results ===
        # If no results, attempt fallback logic such as type reclassification or alternate search strategies
        if not search_results:
            console(f"🚫 No result found for “{search.title}” ({search.year}) [{media_type}]", "RED")
            logger.warning(f"🚫 No result found for “{search.title}” ({search.year}) [{media_type}]")
            # Movie-to-TV fallback for single-file items
            if media_type == "movie" and hasattr(search, "files") and len(search.files) == 1:
                logger.info(f"🔄 Movie lookup failed; retrying as TV series for single-file “{search.title}”")
                console(f"🔄 Retrying as TV series: “{search.title}”", "YELLOW")
                tv_result = query_tmdb(search, "tv_series")
                if tv_result:
                    RECLASSIFIED.append({
                        "original_type": "movie",
                        "new_type": "tv_series",
                        "title": search.title,
                        "year": search.year,
                        "matched_id": getattr(tv_result, "id", None),
                        "file": os.path.basename(search.files[0])
                    })
                    search.type = "tv_series"
                    selected_id = getattr(tv_result, "id", None)
                    return tv_result
            raise NoResultsError(f"No result found for {search.title} ({search.year}) [{media_type}]")

        # === Step 3: Match by Known IDs ===
        # Try to match using TMDB, TVDB, or IMDB IDs if present
        id_match = match_by_id(search_results, search, media_type)
        if id_match:
            search.match_reason = "id"
            return id_match

        # === Step 4: Exact Title + Year Match ===
        # Try for an exact match on normalized title and year
        shortcut = exact_match_shortcut(search_results, search)
        if shortcut:
            header = "🎯 Exact match:"
            msg = f"  → “{search.title}” ({search.year}) [{getattr(shortcut, 'id', None)}] [{media_type}]"
            logger.info(header)
            logger.info(msg)
            console(header)
            console(msg, "GREEN")
            selected_id = getattr(shortcut, 'id', None)
            search.match_reason = "exact"
            return shortcut

        # === Step 5: Original Title Match ===
        # Try matching on the original (non-localized) title
        orig_match = match_by_original_title(search_results, search, media_type)
        if orig_match:
            search.match_reason = "original"
            return orig_match

        # === Step 6: Alternate Titles / Fuzzy Match ===
        # Try alternate title match and high-confidence fuzzy matching
        fallback_match = attempt_fallbacks(search_results, search, media_type)
        if fallback_match:
            # Determine which fallback was used
            if alternate_titles_fallback(search_results, search, media_type) == fallback_match:
                search.match_reason = "alternate"
            else:
                search.match_reason = "fuzzy"
            return fallback_match

        # === Step 6b: Fuzzy fallback for collections ===
        if media_type == "collection":
            candidates, scored = fuzzy_scoring(search_results, search)
            if candidates:
                best = candidates[0]
                header = "🎯 Fuzzy fallback collection match:"
                msg = f"  → {getattr(best, 'name', getattr(best, 'title', ''))} [{getattr(best, 'id', '?')}] [{media_type}]"
                console(header)
                console(msg, "GREEN")
                logger.info(header)
                logger.info(msg)
                search.match_reason = "collection_fuzzy"
                return best
        
        # === Step 7: Final Fallback for Single Match ===
        # If all else fails, accept a single high-similarity result if no year is specified
        if retry and search.year is None and len(search_results) == 1:
            candidate = search_results[0]
            title = getattr(candidate, 'title', getattr(candidate, 'name', ''))
            norm_input = normalize_str(search.title)
            norm_candidate = normalize_str(title)
            ratio = SequenceMatcher(None, norm_input, norm_candidate).ratio()
            if ratio >= 0.95:
                msg = f"🟡 Final fallback accepted single result: {title} [id={getattr(candidate, 'id', '?')}] [{media_type}]"
                console(msg, "YELLOW")
                logger.warning(msg)
                search.match_reason = "fallback_single"
                return candidate

        # If no confident match found after all strategies
        msg = f"🤷 No confident match found for “{search.title}” ({search.year})"
        logger.warning(msg)
        raise NoResultsError(msg)

    except ConnectionError as ce:
        console(f"[ERROR] Connection failed for '{search.title}': {ce}", "RED")
        logger.error(f" Connection failed for '{search.title}': {ce}")
    except NoResultsError as nre:
        logger.warning(str(nre))
        console(f"[WARNING] {str(nre)}", "YELLOW")
    except Exception as e:
        console(f"[WARNING] Failed to query TMDB for '{orig_title}' ({search.year}) as {media_type}: {e}", "YELLOW")
        logger.warning(f"Failed to query TMDB for '{orig_title}' ({search.year}) as {media_type}: {e}")
        if "No Results Found" in str(e):
            # 1) Unaccented fallback: try again with unaccented title
            if not retry_unidecode:
                unaccented = unidecode(orig_title)
                if unaccented != orig_title:
                    console(f"[WARNING] 🔁 Retrying TMDB search with unaccented title: '{unaccented}'", "YELLOW")
                    logger.warning(f"🔁 Retrying with unaccented title: '{unaccented}'")
                    search.title = unaccented
                    return query_tmdb(search, media_type, retry=retry, retry_unidecode=True)
            # 2) Underscore-to-space fallback: try again with underscores replaced by spaces
            if not retry and "_" in orig_title:
                alt_title = orig_title.replace("_", " ")
                console(f"[WARNING] 🔁 Retrying TMDB search with out underscores: '{alt_title}'", "YELLOW")
                logger.warning(f"🔁 Retrying with spaces: '{alt_title}'")
                search.title = alt_title
                return query_tmdb(search, media_type, retry=True, retry_unidecode=retry_unidecode)
            # 3) Hyphen-to-space fallback: try again with hyphens replaced by spaces
            if not retry and "-" in orig_title:
                alt_title = orig_title.replace("-", " ")
                console(f"[WARNING] 🔁 Retrying TMDB search without hyphens: '{alt_title}'", "YELLOW")
                logger.warning(f"🔁 Retrying with spaces: '{alt_title}'")
                search.title = alt_title
                return query_tmdb(search, media_type, retry=True, retry_unidecode=retry_unidecode)
            # 4) Movie-to-TV fallback for single-file items
            if media_type == "movie" and hasattr(search, "files") and len(search.files) == 1:
                logger.info(f"🔄 Movie lookup failed; retrying as TV series for single-file “{search.title}”")
                console(f"🔄 Retrying as TV series: “{search.title}”", "YELLOW")
                tv_result = query_tmdb(search, "tv_series")
                if tv_result:
                    RECLASSIFIED.append({
                        "original_type": "movie",
                        "new_type": "tv_series",
                        "title": search.title,
                        "year": search.year,
                        "matched_id": getattr(tv_result, "id", None),
                        "file": os.path.basename(search.files[0])
                    })
                    search.type = "tv_series"
                    selected_id = getattr(tv_result, "id", None)
                    return tv_result
            # 5) Final fallback: try search with no year if all else failed
            if media_type in ("movie", "tv_series") and not retry:
                logger.warning(f"🔁 Final fallback: retrying TMDB search with no year for “{search.title}”")
                console(f"🔁 Final fallback: retrying TMDB search with no year", "YELLOW")
                search.year = None
                return query_tmdb(search, media_type, retry=True, retry_unidecode=retry_unidecode)

def handle_data(items: List[MediaItem]) -> List[MediaItem]:
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
    if QUIET:
        progress = tqdm(total=len(items), desc="🔍 Enriching metadata", unit="item")
    for item in items:
        enriched = item.enrich()
        if QUIET and progress is not None:
            progress.update(1)
        if not enriched:
            item.match_failed = True
            UNMATCHED_CASES.append({
                "media_type": item.type,
                "title": item.title,
                "year": item.year,
                "tmdb_id": getattr(item, "tmdb_id", ""),
                "tvdb_id": getattr(item, "tvdb_id", ""),
                "imdb_id": getattr(item, "imdb_id", ""),
                "files": ";".join(item.files)
            })
            continue
        if item.type == "tv_series":
            has_tvdb = getattr(item, "tvdb_id", None) or getattr(item, "new_tvdb_id", None)
            if not has_tvdb:
                TVDB_MISSING_CASES.append({
                    "title": item.title,
                    "year": item.year,
                    "tmdb_id": getattr(item, "tmdb_id", ""),
                    "imdb_id": getattr(item, "imdb_id", ""),
                    "files": ";".join(item.files)
                })
    if QUIET and progress is not None:
        progress.close()
    print("✅ Completed metadata enrichment")
    logger.info("✅ Completed metadata enrichment")
    return items

def rename_files(items: List[MediaItem]) -> list:
    """
    Rename files for all enriched MediaItem objects, respecting DRY_RUN mode.
    Handles filename conflicts, length limits, and logs all actions.
    Args:
        items: List of MediaItem instances to process.
    Returns:
        List of tuples (media_type, old_filename, new_filename) for updated files.
    """
    mode = "DRY RUN" if DRY_RUN else "LIVE"
    logger.info(f"🏷  Starting file rename process ({mode} mode)")
    file_updates = []  # collects tuples of (media_type, old_filename, new_filename)
    renamed, skipped = 0, 0
    existing_filenames = {f.lower() for f in os.listdir(source_dir)}
    for media_item in items:
        media_type = media_item.type
        if getattr(media_item, "match_failed", False):
            continue
        # Define a per-item header flag for tv_series DRY_RUN grouping
        header_printed = False
        for index, file_path in enumerate(media_item.files):
            directory, old_filename = os.path.split(file_path)
            new_filename = generate_new_filename(media_item, old_filename)
            if old_filename == new_filename:
                if DEBUG_MODE:
                    console(f"⏭️ Skipping unchanged file: {old_filename}", "YELLOW")
                    logger.debug(f"Skipping unchanged file: {old_filename}")
                skipped += 1
                continue
            base_title = media_item.new_title if media_item.new_title is not None else media_item.title
            base_year  = media_item.new_year   if media_item.new_year   is not None else media_item.year
            id_suffixes = []
            tmdb_id = getattr(media_item, 'new_tmdb_id', getattr(media_item, 'tmdb_id', None))
            tvdb_id = getattr(media_item, 'new_tvdb_id', getattr(media_item, 'tvdb_id', None))
            imdb_id = getattr(media_item, 'new_imdb_id', getattr(media_item, 'imdb_id', None))
            if tmdb_id:
                id_suffixes.append(f"tmdb-{tmdb_id}")
            if tvdb_id:
                id_suffixes.append(f"tvdb-{tvdb_id}")
            if imdb_id:
                id_suffixes.append(f"imdb-{imdb_id}")
            suffix = ' ' + ' '.join(f'{{{id_str}}}' for id_str in id_suffixes) if id_suffixes else ''
            ext = os.path.splitext(old_filename)[-1]
            season_suffix = ''
            match = SEASON_PATTERN.search(old_filename)
            if match:
                season_suffix = match.group(0)
            base = f"{base_title}{f' ({base_year})' if base_year else ''}"
            if media_type == "tv_series":
                new_filename = f"{base}{suffix}{season_suffix}{ext}"
            else:
                new_filename = f"{base}{season_suffix}{suffix}{ext}"
            new_filename = sanitize_filename(new_filename)
            new_filename = ' '.join(new_filename.split()).strip()
            if len(new_filename) > 255:
                console(f"⛔ Skipped (too long): {new_filename}", "RED")
                logger.warning(f"⛔ Skipped (too long): {new_filename}")
                skipped += 1
                continue
            if new_filename.lower() in existing_filenames and old_filename.lower() != new_filename.lower():
                console(f"⚠️  Skipped (name conflict): {new_filename}", "YELLOW")
                logger.warning(f"⚠️  Skipped (name conflict): {new_filename}")
                skipped += 1
                continue
            new_path = os.path.join(directory, new_filename)
            if old_filename == new_filename:
                continue
            if DRY_RUN:
                if media_type == "tv_series":
                    # Print series header once
                    if not header_printed:
                        console(f"[DRY RUN] Series: {base_title} ({base_year})")
                        logger.info(f"[DRY RUN] Series: {base_title} ({base_year})")
                        header_printed = True
                    # Print each file rename as a bullet
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
                    # existing non-tv_series DRY_RUN logic unchanged
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
    # Print renaming header if any files were renamed or debug mode is on
    if file_updates or DEBUG_MODE:
        console(f"📂 Renaming files:")
        logger.info(f"📂 Renaming files:")
    if DRY_RUN:
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
    # Backup of renamed files
    if file_updates:
        backup_path = os.path.join(LOG_DIR, "renamed_backup.json")
        with open(backup_path, "w") as f:
            json.dump([
                {"old": old, "new": new, "type": typ}
                for typ, old, new in file_updates
            ], f, indent=2)
        logger.info(f"📝 Backup of renamed files written to {backup_path}")
    return file_updates

def main():
    global CACHE
    CACHE = load_cache()
    start_time = time.time()
    items = scan_files_in_flat_folder(source_dir)
    updated_items = handle_data(items)
    file_updates = rename_files(updated_items)
    # Export merged metadata+renames CSV in debug mode
   
    csv_path = os.path.join(LOG_DIR, "updated_files.csv")
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # Header row with metadata + filenames + match_reason
        writer.writerow([
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
        ])
        for media_type, old_fn, new_fn in file_updates:
            # Find the matching item in updated_items
            matched = next(
                (item for item in updated_items
                    if item.type == media_type
                    and any(os.path.basename(fp) == old_fn for fp in item.files)),
                None
            )
            original_title = matched.title if matched else ""
            new_title = getattr(matched, "new_title", "") if matched else ""
            original_year = matched.year if matched else ""
            new_year = getattr(matched, "new_year", "") if matched else ""
            tmdb_id = matched.tmdb_id if matched else ""
            new_tmdb_id = getattr(matched, "new_tmdb_id", "") if matched else ""
            tvdb_id = getattr(matched, "tvdb_id", "") if matched else ""
            new_tvdb_id = getattr(matched, "new_tvdb_id", "") if matched else ""
            imdb_id = matched.imdb_id if matched else ""
            new_imdb_id = getattr(matched, "new_imdb_id", "") if matched else ""
            match_reason = getattr(matched, "match_reason", "") if matched else ""
            writer.writerow([
                media_type,
                old_fn,
                new_fn,
                original_title,
                new_title,
                original_year,
                new_year,
                tmdb_id,
                new_tmdb_id,
                tvdb_id,
                new_tvdb_id,
                imdb_id,
                new_imdb_id,
                match_reason,
            ])
    console(f"⚙️ Updated files CSV written to {csv_path}")
    logger.info(f"Updated files CSV written to {csv_path}")

    # Export unmatched cases to CSV
    if UNMATCHED_CASES:
        unmatched_csv = os.path.join(LOG_DIR, "unmatched_items.csv")
        with open(unmatched_csv, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["#", "media_type", "files", "title", "year", "tmdb_id", "tvdb_id", "imdb_id"])
            for idx, case in enumerate(UNMATCHED_CASES, 1):
                writer.writerow([
                    idx,
                    case["media_type"],
                    os.path.basename(case["files"]) if isinstance(case["files"], str) else ";".join(os.path.basename(f) for f in case["files"].split(";")),
                    case["title"],
                    case["year"],
                    case["tmdb_id"],
                    case["tvdb_id"],
                    case["imdb_id"],
                ])
        console(f"⚠️ Unmatched items written to {unmatched_csv}")
        logger.info(f"Unmatched items written to {unmatched_csv}")

    # Export TV series missing TVDB mapping
    if TVDB_MISSING_CASES:
        tvdb_csv = os.path.join(LOG_DIR, "missing_tvdb.csv")
        with open(tvdb_csv, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["title", "year", "tmdb_id", "imdb_id", "files"])
            for case in TVDB_MISSING_CASES:
                writer.writerow([
                    case["title"],
                    case["year"],
                    case["tmdb_id"],
                    case["imdb_id"],
                    ";".join(os.path.basename(f) for f in case["files"].split(";")) if isinstance(case["files"], str) else ""
                ])
        console(f"⚠️ TV series missing TVDB IDs written to {tvdb_csv}")
        logger.warning(f"TV series missing TVDB IDs written to {tvdb_csv}")
    # Performance summary
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
    total_items = len(items)
    total_unmatched = len(UNMATCHED_CASES)
    total_missing_tvdb = len(TVDB_MISSING_CASES)
    total_reclassified = len(RECLASSIFIED)
    total_renames = len(file_updates)

    summary = (
        f"\n🏁 Performance Summary:\n"
        f"\t• Time elapsed: {elapsed_str}s\n"
        f"\t• Items processed: {total_items}\n"
        f"\t• Files renamed: {total_renames}\n"
        f"\t• Unmatched items: {total_unmatched}\n"
        f"\t• TV series missing TVDB: {total_missing_tvdb}\n"
        f"\t• Reclassified (movie→TV): {total_reclassified}\n"
    )
    print(summary)
    logger.info(summary)
    # Construct active_keys set of cache keys for items that still exist
    active_keys = {
        f"{item.title} ({item.year}) [{item.type}]"
        for item in updated_items
    }
    save_cache(CACHE, active_keys)


if __name__ == "__main__":
    main()