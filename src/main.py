#!/usr/bin/env python3
"""
Pokemon Card Scanner
Processes a 3x3 grid of Pokemon cards from a single image.
"""

import argparse
import os
import csv
import json
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import requests
from PIL import Image
import pytesseract
import cv2
import numpy as np
import re


class PokemonCardScanner:
    def __init__(self, offline_mode: bool = False):
        self.offline_mode = offline_mode
        self.api_base_url = "https://api.pokemontcg.io/v2/cards"
        # Rate limiting for API calls
        self.last_api_call = 0
        self.api_delay = 0.1  # 100ms between calls

    def split_image_3x3(self, image_path: str) -> List[Image.Image]:
        """Split a 3x3 grid image into 9 individual card images."""
        try:
            img = Image.open(image_path)
            width, height = img.size

            # Calculate dimensions for each card
            card_width = width // 3
            card_height = height // 3

            cards = []
            for row in range(3):
                for col in range(3):
                    # Calculate crop coordinates
                    left = col * card_width
                    top = row * card_height
                    right = left + card_width
                    bottom = top + card_height

                    # Crop the card
                    card = img.crop((left, top, right, bottom))
                    cards.append(card)

            return cards

        except Exception as e:
            print(f"Error splitting image: {e}")
            return []

    def preprocess_card_image(self, card_image: Image.Image) -> Image.Image:
        """Preprocess card image for better OCR results."""
        # Convert PIL to OpenCV format
        cv_image = cv2.cvtColor(np.array(card_image), cv2.COLOR_RGB2BGR)

        # Resize image for better OCR (OCR works better on larger images)
        height, width = cv_image.shape[:2]
        scale_factor = 2.0
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        cv_image = cv2.resize(cv_image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

        # Convert to grayscale
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        # Noise reduction with bilateral filter (preserves edges better)
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)

        # Multiple thresholding approaches - we'll return the best one
        # Method 1: OTSU
        _, thresh1 = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Method 2: Adaptive threshold
        thresh2 = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 11, 2)

        # Method 3: Simple threshold at a higher value (good for white text on dark)
        _, thresh3 = cv2.threshold(filtered, 150, 255, cv2.THRESH_BINARY)

        # Choose the best threshold (you can experiment with this)
        # For now, let's use OTSU as it's generally robust
        final_thresh = thresh1

        # Morphological operations to clean up the image
        kernel = np.ones((2, 2), np.uint8)
        final_thresh = cv2.morphologyEx(final_thresh, cv2.MORPH_CLOSE, kernel)
        final_thresh = cv2.morphologyEx(final_thresh, cv2.MORPH_OPEN, kernel)

        # Convert back to PIL
        return Image.fromarray(final_thresh)

    def extract_text_from_card(self, card_image: Image.Image) -> str:
        """Extract text from card image using OCR."""
        try:
            # Try multiple preprocessing approaches
            results = []

            # Approach 1: Standard preprocessing
            processed_image1 = self.preprocess_card_image(card_image)
            config1 = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789/- '
            text1 = pytesseract.image_to_string(processed_image1, config=config1).strip()
            results.append(text1)

            # Approach 2: Focus on just the top portion (where card name usually is)
            width, height = card_image.size
            top_crop = card_image.crop((0, 0, width, height // 3))  # Top third
            processed_top = self.preprocess_card_image(top_crop)
            config2 = r'--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz '
            text2 = pytesseract.image_to_string(processed_top, config=config2).strip()
            results.append(text2)

            # Approach 3: Try with original image (sometimes works better)
            config3 = r'--oem 3 --psm 7'
            text3 = pytesseract.image_to_string(card_image, config=config3).strip()
            results.append(text3)

            # Approach 4: Focus on bottom area where set info might be
            bottom_crop = card_image.crop((0, height * 2 // 3, width, height))  # Bottom third
            processed_bottom = self.preprocess_card_image(bottom_crop)
            text4 = pytesseract.image_to_string(processed_bottom, config=config1).strip()
            results.append(text4)

            # Combine results, taking the longest reasonable text
            all_text = "\n".join(results)
            return all_text

        except Exception as e:
            print(f"Error extracting text: {e}")
            return ""

    def parse_card_info(self, ocr_text: str) -> Dict[str, str]:
        """Parse card information from OCR text."""
        card_info = {
            'name': '',
            'number': '',
            'set': '',
            'rarity': '',
            'raw_text': ocr_text
        }

        lines = [line.strip() for line in ocr_text.split('\n') if line.strip()]

        if not lines:
            return card_info

        # Find the most likely card name (usually the cleanest, longest line in the first few lines)
        potential_names = []
        for i, line in enumerate(lines[:5]):  # Look at first 5 lines
            # Clean up the line
            cleaned = re.sub(r'[^A-Za-z\s]', '', line).strip()
            # Skip very short or very long lines, or lines with too many spaces
            if 3 <= len(cleaned) <= 25 and cleaned.count(' ') <= 3:
                potential_names.append((cleaned, len(cleaned)))

        if potential_names:
            # Choose the longest reasonable name
            card_info['name'] = max(potential_names, key=lambda x: x[1])[0]

        # Look for card numbers with better patterns
        number_patterns = [
            r'(\d{1,3})/(\d{1,3})',  # 123/456 format
            r'(\d{1,3})\s*/\s*(\d{1,3})',  # 123 / 456 with spaces
            r'#(\d{1,3})',  # #123 format
            r'No\.?\s*(\d{1,3})',  # No. 123 format
        ]

        for line in lines:
            for pattern in number_patterns:
                match = re.search(pattern, line)
                if match:
                    if len(match.groups()) >= 2 and match.group(2):
                        card_info['number'] = f"{match.group(1)}/{match.group(2)}"
                    else:
                        card_info['number'] = match.group(1)
                    break
            if card_info['number']:
                break

        # Look for set information (common set abbreviations)
        set_patterns = [
            r'\b(BS|JU|FO|TR|GS|LC|AQ|SK|EX|DP|PL|HS|BW|XY|SM|SW|SH)\b',
            r'\b(Base|Jungle|Fossil|Rocket|Gym|Neo|Legend|Aquapolis|Skyridge)\b',
        ]

        for line in lines:
            line_upper = line.upper()
            for pattern in set_patterns:
                match = re.search(pattern, line_upper)
                if match:
                    card_info['set'] = match.group(1)
                    break
            if card_info['set']:
                break

        # Look for rarity indicators
        rarity_patterns = {
            'common': r'\b(COMMON|C)\b',
            'uncommon': r'\b(UNCOMMON|UC|U)\b',
            'rare': r'\b(RARE|R)\b',
            'holo': r'\b(HOLO|HOLOGRAPHIC)\b',
            'ultra': r'\b(ULTRA|UR)\b',
            'secret': r'\b(SECRET|SR)\b',
            'promo': r'\b(PROMO|P)\b'
        }

        for line in lines:
            line_upper = line.upper()
            for rarity, pattern in rarity_patterns.items():
                if re.search(pattern, line_upper):
                    card_info['rarity'] = rarity
                    break
            if card_info['rarity']:
                break

        return card_info

    def get_card_price(self, card_name: str, card_number: str = "") -> Optional[Dict]:
        """Get card price from Pokemon TCG API."""
        if self.offline_mode:
            return None

        try:
            # Rate limiting
            current_time = time.time()
            if current_time - self.last_api_call < self.api_delay:
                time.sleep(self.api_delay)
            self.last_api_call = time.time()

            # Build query
            query = f'name:"{card_name}"'
            if card_number:
                query += f' number:{card_number.split("/")[0]}'

            params = {
                'q': query,
                'select': 'name,number,set,cardmarket'
            }

            response = requests.get(self.api_base_url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()

            if data.get('data') and len(data['data']) > 0:
                card = data['data'][0]  # Take first match

                price_info = {
                    'api_name': card.get('name', ''),
                    'api_number': card.get('number', ''),
                    'api_set': card.get('set', {}).get('name', ''),
                    'market_price': 0.0,
                    'low_price': 0.0,
                    'high_price': 0.0
                }

                # Extract price information
                cardmarket = card.get('cardmarket', {})
                if cardmarket:
                    prices = cardmarket.get('prices', {})
                    price_info['market_price'] = prices.get('averageSellPrice', 0.0) or 0.0
                    price_info['low_price'] = prices.get('lowPrice', 0.0) or 0.0
                    price_info['high_price'] = prices.get('highPrice', 0.0) or 0.0

                return price_info

        except requests.exceptions.RequestException as e:
            print(f"API request failed for {card_name}: {e}")
        except Exception as e:
            print(f"Error getting price for {card_name}: {e}")

        return None

    def process_cards(self, image_path: str, output_dir: str) -> List[Dict]:
        """Process all cards from the input image."""
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Split image into individual cards
        print("Splitting image into individual cards...")
        cards = self.split_image_3x3(image_path)

        if not cards:
            print("Failed to split image")
            return []

        results = []

        for i, card_image in enumerate(cards, 1):
            print(f"Processing card {i}/9...")

            # Save cropped card image
            card_filename = f"card_{i:02d}.jpg"
            card_path = os.path.join(output_dir, card_filename)
            card_image.save(card_path, 'JPEG', quality=95)

            # Extract text using OCR
            ocr_text = self.extract_text_from_card(card_image)

            # Parse card information
            card_info = self.parse_card_info(ocr_text)

            # Add metadata
            card_info['card_number'] = i
            card_info['image_file'] = card_filename

            # Get price information if in online mode
            if not self.offline_mode and card_info['name']:
                print(f"  Getting price for: {card_info['name']}")
                price_info = self.get_card_price(card_info['name'], card_info['number'])
                if price_info:
                    card_info.update(price_info)
                else:
                    # Add empty price fields
                    card_info.update({
                        'api_name': '',
                        'api_number': '',
                        'api_set': '',
                        'market_price': 0.0,
                        'low_price': 0.0,
                        'high_price': 0.0
                    })

            results.append(card_info)

        return results

    def save_to_csv(self, card_data: List[Dict], output_path: str):
        """Save card data to CSV file."""
        if not card_data:
            print("No card data to save")
            return

        # Define CSV headers based on available data
        headers = ['card_number', 'image_file', 'name', 'number', 'set', 'rarity']

        if not self.offline_mode:
            headers.extend(['api_name', 'api_number', 'api_set', 'market_price', 'low_price', 'high_price'])

        headers.append('raw_text')

        try:
            with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=headers)
                writer.writeheader()

                for card in card_data:
                    # Ensure all fields exist
                    row = {header: card.get(header, '') for header in headers}
                    writer.writerow(row)

            print(f"CSV saved to: {output_path}")

        except Exception as e:
            print(f"Error saving CSV: {e}")


def main():
    parser = argparse.ArgumentParser(description='Pokemon Card Scanner')
    parser.add_argument('input_image', help='Path to input image (3x3 grid of Pokemon cards)')
    parser.add_argument('output_dir', help='Output directory for cropped images and CSV')
    parser.add_argument('--offline', action='store_true',
                        help='Run in offline mode (no API calls for pricing)')

    args = parser.parse_args()

    # Validate input file
    if not os.path.exists(args.input_image):
        print(f"Error: Input image '{args.input_image}' not found")
        return 1

    # Initialize scanner
    scanner = PokemonCardScanner(offline_mode=args.offline)

    print(f"Processing image: {args.input_image}")
    print(f"Output directory: {args.output_dir}")
    print(f"Mode: {'Offline' if args.offline else 'Online'}")

    # Process cards
    card_data = scanner.process_cards(args.input_image, args.output_dir)

    if card_data:
        # Save results to CSV
        csv_path = os.path.join(args.output_dir, 'pokemon_cards.csv')
        scanner.save_to_csv(card_data, csv_path)

        print(f"\nProcessing complete!")
        print(f"Processed {len(card_data)} cards")
        print(f"Images saved to: {args.output_dir}")
        print(f"CSV saved to: {csv_path}")

        # Show summary
        if not args.offline:
            total_value = sum(card.get('market_price', 0) for card in card_data)
            print(f"Total estimated value: ${total_value:.2f}")
    else:
        print("No cards processed successfully")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())