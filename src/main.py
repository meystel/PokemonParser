#!/usr/bin/env python3
import argparse
import os
import csv
import json
import warnings
from typing import List, Dict
from PIL import Image
import cv2
import torch
import easyocr
import re
from card_name_resolver import CardNameResolver

# Suppress noisy warnings
warnings.filterwarnings("ignore", message=".*pin_memory.*")

class PokemonImageScanner:
    def __init__(self, offline_mode: bool = False, api_key: str = None):
        self.offline_mode = offline_mode
        use_mps = torch.backends.mps.is_available()
        self.reader = easyocr.Reader(['en'], gpu=use_mps)
        if use_mps:
            print("✅ Using Apple GPU (MPS) for OCR")
        else:
            print("⚠️ MPS not available, using CPU")
        self.card_database = {}
        self.load_card_database()
        self.resolver = CardNameResolver(api_key=api_key, offline=offline_mode)

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

    def _ocr_region(self, image, y1_ratio, y2_ratio, x1_ratio, x2_ratio):
        h, w, _ = image.shape
        y1 = int(h * y1_ratio)
        y2 = int(h * y2_ratio)
        x1 = int(w * x1_ratio)
        x2 = int(w * x2_ratio)
        region = image[y1:y2, x1:x2]

        # Preprocess: grayscale + adaptive thresholding (good for shiny backgrounds)
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )

        # Run OCR
        results = self.reader.readtext(thresh)

        # Keep confidence filter at 0.4 to reduce junk
        return [txt for (_, txt, conf) in results if conf > 0.4]

    def _clean_name(self, raw_name: str) -> str:
        name = raw_name
        name = name.replace("VzX", "V").replace("Vzax", "V")
        name = re.sub(r"Ggantamax", "Gigantamax", name, flags=re.IGNORECASE)
        name = name.replace("@", "").strip()
        name = re.sub(r"\s+", " ", name)
        name = " ".join([w.capitalize() for w in name.split()])
        return name

    def extract_card_text(self, card_image_path: str) -> dict:
        img_cv = cv2.imread(card_image_path)
        if img_cv is None:
            return {"name": "", "card_number": "", "hp": ""}

        # OCR for card name (tighter crop, just under top border)
        name_texts = self._ocr_region(img_cv, 0.055, 0.125, 0.10, 0.85)
        raw_name = " ".join(name_texts).strip()
        name = self._clean_name(raw_name)
        name = self.resolver.resolve(name)

        # OCR for card number (bottom-right tighter window)
        num_texts = self._ocr_region(img_cv, 0.90, 0.97, 0.65, 0.93)
        card_number = ""
        for t in num_texts:
            if "/" in t:
                card_number = t.strip()
                break

        # OCR for HP (top-right)
        hp_texts = self._ocr_region(img_cv, 0.045, 0.11, 0.75, 0.95)
        hp = ""
        for t in hp_texts:
            if t.isdigit():
                hp = t
                break

        return {
            "name": name,
            "card_number": card_number,
            "hp": hp
        }

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
                "hp": ocr_data["hp"],
                "set_name": "",
                "card_type": "",
                "market_price": "",
                "similarity_score": "",
                "perceptual_hash": "",
                "edge_density": "",
                "dominant_colors": ""
            }

            if not self.offline_mode and ocr_data["name"]:
                details = self.resolver.fetch_card_details(ocr_data["name"], ocr_data["card_number"])
                if details:
                    card_info["name"] = details.get("name", card_info["name"])
                    card_info["hp"] = details.get("hp", card_info["hp"])
                    card_info["set_name"] = details.get("set", {}).get("name", "")
                    card_info["card_type"] = ", ".join(details.get("types", []))
                    prices = details.get("tcgplayer", {}).get("prices", {})
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
    parser = argparse.ArgumentParser(description="Pokemon Card Scanner with EasyOCR + Resolver")
    parser.add_argument("input_image", help="Path to input image (3x3 grid)")
    parser.add_argument("output_dir", help="Output directory")
    parser.add_argument("--offline", action="store_true", help="Run without API")
    parser.add_argument("--api_key", help="Pokémon TCG API key (optional)")
    args = parser.parse_args()

    scanner = PokemonImageScanner(offline_mode=args.offline, api_key=args.api_key)
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
