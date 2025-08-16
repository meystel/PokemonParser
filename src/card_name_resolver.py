import re
import time
import json
import os
import requests
from rapidfuzz import process, fuzz

API_BASE = "https://api.pokemontcg.io/v2/cards"
CACHE_FILE = os.path.join(os.path.dirname(__file__), "card_names_cache.json")

class CardNameResolver:
    def __init__(self, api_key: str = None, offline: bool = False):
        self.api_key = api_key
        self.offline = offline
        self.card_names = []
        if not offline:
            self._load_all_card_names()

    def _load_all_card_names(self):
        # Try to load from cache first
        if os.path.exists(CACHE_FILE):
            try:
                with open(CACHE_FILE, "r", encoding="utf-8") as f:
                    self.card_names = json.load(f)
                print(f"Loaded {len(self.card_names)} card names from cache.")
                return
            except Exception as e:
                print(f"Cache load failed, refetching from API: {e}")

        print("Fetching card list from Pokémon TCG API...")
        headers = {"X-Api-Key": self.api_key} if self.api_key else {}
        page = 1
        while True:
            resp = requests.get(
                API_BASE, params={"page": page, "pageSize": 250, "select": "name"},
                headers=headers
            )
            if resp.status_code != 200:
                break
            data = resp.json().get("data", [])
            if not data:
                break
            self.card_names.extend([c["name"] for c in data])
            print(f"  → Loaded {len(self.card_names)} names so far")
            page += 1
            time.sleep(0.2)

        if self.card_names:
            try:
                with open(CACHE_FILE, "w", encoding="utf-8") as f:
                    json.dump(self.card_names, f)
                print(f"Saved {len(self.card_names)} card names to cache.")
            except Exception as e:
                print(f"Failed to save cache: {e}")

        print(f"Total cards loaded: {len(self.card_names)}")

    @staticmethod
    def normalize(text: str) -> str:
        """Normalize OCR text by lowercasing and stripping unwanted chars."""
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", "", text)  # allow digits too (helps EX, VSTAR, etc.)
        return text.strip()

    def resolve(self, ocr_text: str, threshold: int = 70) -> str:
        """Resolve OCR text to closest known card name."""
        clean = self.normalize(ocr_text)
        if not clean or self.offline:
            return ocr_text  # fallback: keep OCR as-is if offline
        result = process.extractOne(clean, self.card_names, scorer=fuzz.WRatio)
        if result:
            match, score, _ = result
            return match if score >= threshold else ocr_text
        return ocr_text

    def resolve_candidates(self, candidates: list, threshold: int = 60) -> str:
        """Resolve a list of OCR candidate strings to the best possible card name."""
        if not candidates:
            return "Unknown"

        # In offline mode: just pick the *longest candidate* (OCR often has multiple variants)
        if not self.card_names:
            return max(candidates, key=lambda c: len(c.strip()), default="Unknown").strip()

        best_score = 0
        best_match = "Unknown"

        for cand in candidates:
            norm = self.normalize(cand)
            if not norm:
                continue

            result = process.extractOne(norm, self.card_names, scorer=fuzz.WRatio)
            if result:
                match, score, _ = result
                # Add a slight preference for longer names
                score += len(norm) * 0.2
                if match and score >= threshold and score > best_score:
                    best_score = score
                    best_match = match

        return best_match

    def fetch_card_details(self, card_name: str, card_number: str = None) -> dict:
        """Fetch full card details (set, type, hp, price, etc.) for a resolved name and optional number."""
        if self.offline or not card_name:
            return {}
        headers = {"X-Api-Key": self.api_key} if self.api_key else {}

        # Build query: prioritize exact name + number match if available
        if card_number:
            # Extract the numeric part before "/" if present
            number_part = card_number.split("/")[0]
            query = f'name:"{card_name}" number:"{number_part}"'
        else:
            query = f'name:"{card_name}"'

        resp = requests.get(API_BASE, params={"q": query}, headers=headers)
        if resp.status_code == 200:
            data = resp.json().get("data", [])
            if data:
                return data[0]
        return {}
