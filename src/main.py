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

    def extract_card_text(self, card_image_path: str) -> dict:
        """Extract Pokémon name + suffix and card number from image."""
        img_cv = cv2.imread(card_image_path)
        if img_cv is None:
            return {"name": "", "card_number": ""}

        height, width, _ = img_cv.shape

        # --- Name area (taller crop to catch suffix logos) ---
        y1 = int(height * 0.045)
        y2 = int(height * 0.16)
        x1 = int(width * 0.08)
        x2 = int(width * 0.92)
        name_region = img_cv[y1:y2, x1:x2]
        name_results = self.reader.readtext(name_region)
        name_texts = [txt for (_, txt, conf) in name_results if conf > 0.4]
        name = " ".join(name_texts).strip()

        # --- Card number area (bottom right) ---
        y1 = int(height * 0.87)
        y2 = int(height * 0.96)
        x1 = int(width * 0.60)
        x2 = int(width * 0.92)
        num_region = img_cv[y1:y2, x1:x2]
        num_results = self.reader.readtext(num_region)
        num_texts = [txt for (_, txt, conf) in num_results if conf > 0.4]
        card_number = ""
        for t in num_texts:
            if "/" in t:
                card_number = t.strip()
                break

        return {
            "name": name,
            "card_number": card_number
        }

    def query_card_api(self, query: str) -> dict:
        """Query the Pokémon TCG API using a custom query string."""
        elapsed = time.time() - self.last_api_call
        if elapsed < self.api_delay:
            time.sleep(self.api_delay - elapsed)
        self.last_api_call = time.time()

        url = f"{self.api_base_url}?q={query}"
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
            ocr_data = self.extract_card_text(card_path)

            card_info = {
                "card_number": ocr_data["card_number"],
                "image_file": os.path.basename(card_path),
                "name": ocr_data["name"] or "Unknown",
                "set_name": "",
                "card_type": "",
                "hp": "",
                "market_price": "",
                "similarity_score": "",
                "perceptual_hash": "",
                "edge_density": "",
                "dominant_colors": ""
            }

            if not self.offline_mode and ocr_data["name"]:
                # Build query string
                query_parts = [f'name:"{ocr_data["name"]}"']
                if ocr_data["card_number"]:
                    query_parts.append(f'number:"{ocr_data["card_number"].split("/")[0]}"')
                query_str = " ".join(query_parts)

                api_data = self.query_card_api(query_str)
                if api_data:
                    card_info["set_name"] = api_data.get("set", {}).get("name", "")
                    card_info["card_type"] = ", ".join(api_data.get("types", []))
                    card_info["hp"] = api_data.get("hp", "")
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
    parser = argparse.ArgumentParser(description="Pokemon Card Scanner with EasyOCR (Dual OCR)")
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
