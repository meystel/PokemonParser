#!/usr/bin/env python3
import argparse
import os
import csv
import json
import time
import requests
from pathlib import Path
from typing import List, Dict
from PIL import Image
import cv2
import numpy as np
import easyocr

class PokemonImageScanner:
    def __init__(self, offline_mode: bool = False):
        self.offline_mode = offline_mode
        self.api_base_url = "https://api.pokemontcg.io/v2/cards"
        self.last_api_call = 0
        self.api_delay = 0.5
        self.reader = easyocr.Reader(['en'], gpu=False)
        self.card_database = {}
        self.load_card_database()

    def load_card_database(self):
        if os.path.exists("card_database.json"):
            try:
                with open("card_database.json", "r") as f:
                    self.card_database = json.load(f)
                print(f"Loaded {len(self.card_database)} cards from local database")
            except Exception as e:
                print(f"Error loading card database: {e}")

    def save_card_database(self):
        try:
            with open("card_database.json", "w") as f:
                json.dump(self.card_database, f, indent=2)
        except Exception as e:
            print(f"Error saving card database: {e}")

    def split_image_3x3(self, image_path: str, output_dir: str) -> List[str]:
        img = Image.open(image_path)
        width, height = img.size
        card_width = width // 3
        card_height = height // 3

        os.makedirs(output_dir, exist_ok=True)
        card_paths = []
        card_num = 1
        for row in range(3):
            for col in range(3):
                left = col * card_width
                upper = row * card_height
                right = left + card_width
                lower = upper + card_height
                card_img = img.crop((left, upper, right, lower))
                card_path = os.path.join(output_dir, f"card_{card_num:02d}.jpg")
                card_img.save(card_path)
                card_paths.append(card_path)
                card_num += 1
        return card_paths

    def extract_card_name(self, card_image_path: str) -> str:
        img_cv = cv2.imread(card_image_path)
        if img_cv is None:
            return ""

        height, width, _ = img_cv.shape
        y1 = int(height * 0.05)
        y2 = int(height * 0.14)
        x1 = int(width * 0.12)
        x2 = int(width * 0.88)
        nameplate_region = img_cv[y1:y2, x1:x2]

        results = self.reader.readtext(nameplate_region)
        if not results:
            return ""
        results.sort(key=lambda r: r[2], reverse=True)
        best_text = results[0][1]
        return best_text.strip().replace("\n", " ")

    def query_card_api_by_name(self, card_name: str) -> dict:
        if not card_name:
            return {}
        # Respect rate limiting
        elapsed = time.time() - self.last_api_call
        if elapsed < self.api_delay:
            time.sleep(self.api_delay - elapsed)
        self.last_api_call = time.time()

        url = f"{self.api_base_url}?q=name:\"{card_name}\""
        r = requests.get(url)
        if r.status_code == 200:
            data = r.json().get("data", [])
            if data:
                return data[0]
        return {}

    def process_cards(self, image_path: str, output_dir: str) -> List[Dict]:
        card_paths = self.split_image_3x3(image_path, output_dir)
        all_cards = []
        for idx, card_path in enumerate(card_paths, start=1):
            print(f"Processing {card_path}...")
            card_name = self.extract_card_name(card_path)
            card_info = {
                "card_number": idx,
                "image_file": os.path.basename(card_path),
                "name": card_name or "Unknown",
                "set_name": "",
                "card_type": "",
                "hp": "",
                "market_price": "",
                "similarity_score": "",
                "perceptual_hash": "",
                "edge_density": "",
                "dominant_colors": ""
            }
            if not self.offline_mode and card_name:
                api_data = self.query_card_api_by_name(card_name)
                if api_data:
                    card_info["set_name"] = api_data.get("set", {}).get("name", "")
                    card_info["card_type"] = ", ".join(api_data.get("types", []))
                    card_info["hp"] = api_data.get("hp", "")
                    # Price
                    prices = api_data.get("tcgplayer", {}).get("prices", {})
                    if prices:
                        first_price = list(prices.values())[0]
                        card_info["market_price"] = first_price.get("market", "")
            all_cards.append(card_info)
        return all_cards

    def save_to_csv(self, card_data: List[Dict], output_path: str):
        if not card_data:
            print("No card data to save")
            return
        headers = list(card_data[0].keys())
        with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            for row in card_data:
                writer.writerow(row)
        print(f"CSV saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Pokemon Card Scanner with EasyOCR")
    parser.add_argument("input_image", help="Path to input image (3x3 grid)")
    parser.add_argument("output_dir", help="Output directory")
    parser.add_argument("--offline", action="store_true", help="Run without API")
    args = parser.parse_args()

    scanner = PokemonImageScanner(offline_mode=args.offline)
    print(f"Processing: {args.input_image} (Offline: {args.offline})")
    card_data = scanner.process_cards(args.input_image, args.output_dir)
    if card_data:
        csv_path = os.path.join(args.output_dir, "pokemon_cards_matched.csv")
        scanner.save_to_csv(card_data, csv_path)
        print(f"Processed {len(card_data)} cards.")
    else:
        print("No cards processed.")

if __name__ == "__main__":
    main()
