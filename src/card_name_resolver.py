import re
import time
import requests
from rapidfuzz import process, fuzz

API_BASE = "https://api.pokemontcg.io/v2/cards"

class CardNameResolver:
    def __init__(self, api_key: str = None, offline: bool = False):
        self.api_key = api_key
        self.offline = offline
        self.card_names = []
        if not offline:
            self._load_all_card_names()

    def _load_all_card_names(self):
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
        match, score, _ = process.extractOne(clean, self.card_names, scorer=fuzz.WRatio)
        return match if score >= threshold else ocr_text

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
