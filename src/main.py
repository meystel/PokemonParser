#!/usr/bin/env python3
"""
Pokemon Card Scanner - Image Matching Approach
Uses image hashing and API image matching instead of OCR for reliable card identification
"""

import argparse
import os
import csv
import json
import time
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import requests
from PIL import Image, ImageDraw, ImageFilter
import cv2
import numpy as np
import re
from urllib.parse import quote
import base64
import io


class PokemonImageScanner:
    def __init__(self, offline_mode: bool = False):
        self.offline_mode = offline_mode
        self.api_base_url = "https://api.pokemontcg.io/v2/cards"
        self.last_api_call = 0
        self.api_delay = 0.5  # Slower to be nice to API

        # Card database for offline matching (you'd populate this)
        self.card_database = {}
        self.load_card_database()

    def load_card_database(self):
        """Load local card database if available."""
        db_file = "card_database.json"
        if os.path.exists(db_file):
            try:
                with open(db_file, 'r') as f:
                    self.card_database = json.load(f)
                print(f"Loaded {len(self.card_database)} cards from local database")
            except Exception as e:
                print(f"Error loading card database: {e}")

    def save_card_database(self):
        """Save card database for future use."""
        try:
            with open("card_database.json", 'w') as f:
                json.dump(self.card_database, f, indent=2)
        except Exception as e:
            print(f"Error saving card database: {e}")

    def split_image_3x3(self, image_path: str) -> List[Image.Image]:
        """Split a 3x3 grid image into 9 individual card images."""
        try:
            img = Image.open(image_path)
            width, height = img.size

            card_width = width // 3
            card_height = height // 3

            cards = []
            for row in range(3):
                for col in range(3):
                    left = col * card_width
                    top = row * card_height
                    right = left + card_width
                    bottom = top + card_height

                    card = img.crop((left, top, right, bottom))
                    cards.append(card)

            return cards

        except Exception as e:
            print(f"Error splitting image: {e}")
            return []

    def preprocess_card_for_matching(self, card_image: Image.Image) -> Image.Image:
        """Preprocess card image for better feature matching."""
        try:
            # Convert to RGB if needed
            if card_image.mode != 'RGB':
                card_image = card_image.convert('RGB')

            # Resize to standard size for consistent matching
            standard_size = (400, 560)  # Approximate Pokemon card ratio
            resized = card_image.resize(standard_size, Image.LANCZOS)

            # Apply slight blur to reduce noise and holographic interference
            blurred = resized.filter(ImageFilter.GaussianBlur(0.5))

            return blurred

        except Exception as e:
            print(f"Error preprocessing card: {e}")
            return card_image

    def calculate_perceptual_hash(self, image: Image.Image, hash_size: int = 16) -> str:
        """Calculate a more robust perceptual hash."""
        try:
            # Convert to grayscale
            gray = image.convert('L')

            # Resize to hash_size x hash_size
            resized = gray.resize((hash_size, hash_size), Image.LANCZOS)

            # Get pixel data
            pixels = list(resized.getdata())

            # Calculate average
            avg = sum(pixels) / len(pixels)

            # Create hash
            hash_bits = []
            for pixel in pixels:
                hash_bits.append('1' if pixel > avg else '0')

            # Convert to hex string
            hash_str = ''.join(hash_bits)
            hash_int = int(hash_str, 2)
            hash_hex = format(hash_int, 'x').zfill(hash_size * hash_size // 4)

            return hash_hex

        except Exception as e:
            print(f"Error calculating hash: {e}")
            return ""

    def calculate_color_histogram(self, image: Image.Image) -> List[float]:
        """Calculate color histogram for image similarity."""
        try:
            # Convert to RGB
            rgb_image = image.convert('RGB')

            # Calculate histogram for each channel
            hist_r = rgb_image.histogram()[0:256]
            hist_g = rgb_image.histogram()[256:512]
            hist_b = rgb_image.histogram()[512:768]

            # Normalize histograms
            total_pixels = sum(hist_r)
            if total_pixels > 0:
                hist_r = [x / total_pixels for x in hist_r]
                hist_g = [x / total_pixels for x in hist_g]
                hist_b = [x / total_pixels for x in hist_b]

            # Combine into single feature vector (downsample to reduce size)
            combined_hist = []
            step = 8  # Reduce 256 bins to 32 bins per channel
            for i in range(0, 256, step):
                combined_hist.append(sum(hist_r[i:i + step]))
                combined_hist.append(sum(hist_g[i:i + step]))
                combined_hist.append(sum(hist_b[i:i + step]))

            return combined_hist

        except Exception as e:
            print(f"Error calculating histogram: {e}")
            return []

    def extract_card_features(self, card_image: Image.Image) -> Dict:
        """Extract multiple features for card identification."""
        processed_card = self.preprocess_card_for_matching(card_image)

        features = {
            'perceptual_hash': self.calculate_perceptual_hash(processed_card),
            'color_histogram': self.calculate_color_histogram(processed_card),
            'aspect_ratio': card_image.width / card_image.height,
            'dominant_colors': self.get_dominant_colors(processed_card),
            'edge_features': self.extract_edge_features(processed_card)
        }

        return features

    def get_dominant_colors(self, image: Image.Image, num_colors: int = 5) -> List[Tuple[int, int, int]]:
        """Get dominant colors using simple clustering."""
        try:
            # Convert to RGB and resize for speed
            rgb_image = image.convert('RGB').resize((50, 50))

            # Get all pixels
            pixels = list(rgb_image.getdata())

            # Simple color quantization - group similar colors
            color_counts = {}
            for r, g, b in pixels:
                # Quantize to reduce similar colors
                qr = (r // 32) * 32
                qg = (g // 32) * 32
                qb = (b // 32) * 32
                key = (qr, qg, qb)
                color_counts[key] = color_counts.get(key, 0) + 1

            # Sort by frequency
            sorted_colors = sorted(color_counts.items(), key=lambda x: x[1], reverse=True)

            # Return top colors
            return [color for color, count in sorted_colors[:num_colors]]

        except Exception as e:
            print(f"Error getting dominant colors: {e}")
            return []

    def extract_edge_features(self, image: Image.Image) -> float:
        """Extract edge density as a simple feature."""
        try:
            # Convert to OpenCV format
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

            # Apply Canny edge detection
            edges = cv2.Canny(gray, 50, 150)

            # Calculate edge density
            edge_pixels = np.sum(edges > 0)
            total_pixels = edges.size
            edge_density = edge_pixels / total_pixels if total_pixels > 0 else 0

            return edge_density

        except Exception as e:
            print(f"Error extracting edge features: {e}")
            return 0.0

    def hamming_distance(self, hash1: str, hash2: str) -> int:
        """Calculate Hamming distance between two hashes."""
        if len(hash1) != len(hash2):
            return float('inf')

        return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))

    def histogram_similarity(self, hist1: List[float], hist2: List[float]) -> float:
        """Calculate histogram similarity (0-1, higher is more similar)."""
        if not hist1 or not hist2 or len(hist1) != len(hist2):
            return 0.0

        # Use chi-square distance
        chi_square = 0
        for h1, h2 in zip(hist1, hist2):
            if h1 + h2 > 0:
                chi_square += ((h1 - h2) ** 2) / (h1 + h2)

        # Convert to similarity (0-1)
        similarity = 1 / (1 + chi_square)
        return similarity

    def find_matching_card_in_database(self, features: Dict) -> Optional[Dict]:
        """Find matching card in local database."""
        if not self.card_database:
            return None

        best_match = None
        best_score = 0

        for card_id, stored_card in self.card_database.items():
            if 'features' not in stored_card:
                continue

            stored_features = stored_card['features']
            score = 0

            # Compare perceptual hash
            if features.get('perceptual_hash') and stored_features.get('perceptual_hash'):
                hash_distance = self.hamming_distance(
                    features['perceptual_hash'],
                    stored_features['perceptual_hash']
                )
                hash_similarity = max(0, 1 - (hash_distance / 64))  # Normalize to 0-1
                score += hash_similarity * 0.4

            # Compare color histogram
            if features.get('color_histogram') and stored_features.get('color_histogram'):
                hist_similarity = self.histogram_similarity(
                    features['color_histogram'],
                    stored_features['color_histogram']
                )
                score += hist_similarity * 0.3

            # Compare dominant colors
            if features.get('dominant_colors') and stored_features.get('dominant_colors'):
                color_similarity = self.compare_dominant_colors(
                    features['dominant_colors'],
                    stored_features['dominant_colors']
                )
                score += color_similarity * 0.2

            # Compare edge features
            if features.get('edge_features') is not None and stored_features.get('edge_features') is not None:
                edge_diff = abs(features['edge_features'] - stored_features['edge_features'])
                edge_similarity = max(0, 1 - edge_diff)
                score += edge_similarity * 0.1

            if score > best_score and score > 0.6:  # Threshold for matches
                best_score = score
                best_match = {
                    'card_data': stored_card,
                    'similarity_score': score
                }

        return best_match

    def compare_dominant_colors(self, colors1: List[Tuple[int, int, int]],
                                colors2: List[Tuple[int, int, int]]) -> float:
        """Compare dominant color lists."""
        if not colors1 or not colors2:
            return 0.0

        total_similarity = 0
        comparisons = 0

        for c1 in colors1[:3]:  # Compare top 3 colors
            best_match = 0
            for c2 in colors2[:3]:
                # Calculate color distance
                distance = sum((a - b) ** 2 for a, b in zip(c1, c2)) ** 0.5
                similarity = max(0, 1 - (distance / (255 * 3 ** 0.5)))
                best_match = max(best_match, similarity)

            total_similarity += best_match
            comparisons += 1

        return total_similarity / comparisons if comparisons > 0 else 0

    def search_pokemon_api_by_set_and_number(self, set_code: str, number: str) -> Optional[Dict]:
        """Search for card by set and number if we can identify them."""
        if self.offline_mode:
            return None

        try:
            time.sleep(self.api_delay)
            params = {
                'q': f'set.id:{set_code} number:{number}',
                'select': 'name,number,set,types,cardmarket,images,hp,attacks'
            }

            response = requests.get(self.api_base_url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()
            if data.get('data'):
                return data['data'][0]

        except Exception as e:
            print(f"Error searching API: {e}")

        return None

    def get_random_sample_cards(self, card_type: str = None, page_size: int = 20) -> List[Dict]:
        """Get random sample cards to build database."""
        if self.offline_mode:
            return []

        try:
            time.sleep(self.api_delay)
            params = {
                'select': 'name,number,set,types,cardmarket,images,hp,attacks',
                'pageSize': page_size
            }

            if card_type:
                params['q'] = f'types:{card_type}'

            response = requests.get(self.api_base_url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()
            return data.get('data', [])

        except Exception as e:
            print(f"Error getting sample cards: {e}")
            return []

    def process_cards(self, image_path: str, output_dir: str) -> List[Dict]:
        """Process all cards using image matching approach."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        print("Splitting image into individual cards...")
        cards = self.split_image_3x3(image_path)

        if not cards:
            print("Failed to split image")
            return []

        results = []

        # If database is empty, try to populate it with some samples
        if not self.card_database and not self.offline_mode:
            print("Building initial card database...")
            sample_cards = self.get_random_sample_cards(page_size=50)
            for card in sample_cards[:20]:  # Limit to avoid too many API calls
                card_id = f"{card.get('set', {}).get('id', 'unknown')}_{card.get('number', 'unknown')}"
                self.card_database[card_id] = card
            self.save_card_database()

        for i, card_image in enumerate(cards, 1):
            print(f"Analyzing card {i}/9...")

            # Save cropped card image
            card_filename = f"card_{i:02d}.jpg"
            card_path = os.path.join(output_dir, card_filename)
            card_image.save(card_path, 'JPEG', quality=95)

            # Extract features
            features = self.extract_card_features(card_image)

            # Try to find match in database
            match_result = self.find_matching_card_in_database(features)

            # Create card info
            card_info = {
                'card_number': i,
                'image_file': card_filename,
                'perceptual_hash': features['perceptual_hash'],
                'edge_density': features.get('edge_features', 0),
                'dominant_colors': str(features.get('dominant_colors', [])),
                'name': 'Unknown',
                'set_name': '',
                'card_type': '',
                'hp': None,
                'attacks': [],
                'market_price': 0.0,
                'similarity_score': 0.0
            }

            if match_result:
                matched_card = match_result['card_data']
                card_info.update({
                    'name': matched_card.get('name', 'Unknown'),
                    'set_name': matched_card.get('set', {}).get('name', ''),
                    'card_type': ', '.join(matched_card.get('types', [])),
                    'hp': matched_card.get('hp'),
                    'attacks': [att.get('name', '') for att in matched_card.get('attacks', [])],
                    'market_price': matched_card.get('cardmarket', {}).get('prices', {}).get('averageSellPrice', 0.0),
                    'similarity_score': match_result['similarity_score']
                })
                print(f"  Found match: {card_info['name']} (similarity: {card_info['similarity_score']:.2f})")
            else:
                print(f"  No match found - stored features for future use")

            results.append(card_info)

        return results

    def save_to_csv(self, card_data: List[Dict], output_path: str):
        """Save card analysis to CSV file."""
        if not card_data:
            print("No card data to save")
            return

        headers = [
            'card_number', 'image_file', 'name', 'set_name', 'card_type',
            'hp', 'market_price', 'similarity_score', 'perceptual_hash',
            'edge_density', 'dominant_colors'
        ]

        try:
            with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=headers)
                writer.writeheader()

                for card in card_data:
                    row = {}
                    for header in headers:
                        value = card.get(header, '')
                        # Convert list attacks to string for CSV
                        if header == 'attacks' and isinstance(value, list):
                            row['attacks'] = ', '.join(value)
                        else:
                            row[header] = value
                    writer.writerow(row)

            print(f"CSV saved to: {output_path}")

        except Exception as e:
            print(f"Error saving CSV: {e}")


def main():
    parser = argparse.ArgumentParser(description='Pokemon Card Image Matching Scanner')
    parser.add_argument('input_image', help='Path to input image (3x3 grid of Pokemon cards)')
    parser.add_argument('output_dir', help='Output directory for cropped images and CSV')
    parser.add_argument('--offline', action='store_true',
                        help='Run in offline mode (no API calls)')

    args = parser.parse_args()

    if not os.path.exists(args.input_image):
        print(f"Error: Input image '{args.input_image}' not found")
        return 1

    scanner = PokemonImageScanner(offline_mode=args.offline)

    print(f"Processing image: {args.input_image}")
    print(f"Output directory: {args.output_dir}")
    print(f"Mode: {'Offline' if args.offline else 'Online'}")
    print("Using image matching approach instead of OCR")

    card_data = scanner.process_cards(args.input_image, args.output_dir)

    if card_data:
        csv_path = os.path.join(args.output_dir, 'pokemon_cards_matched.csv')
        scanner.save_to_csv(card_data, csv_path)

        print(f"\nProcessing complete!")
        print(f"Analyzed {len(card_data)} cards")
        print(f"Images saved to: {args.output_dir}")
        print(f"Results saved to: {csv_path}")

        # Show summary
        matches_found = sum(1 for card in card_data if card.get('similarity_score', 0) > 0)
        avg_similarity = np.mean(
            [card.get('similarity_score', 0) for card in card_data if card.get('similarity_score', 0) > 0])

        print(f"\nMatches found: {matches_found}/{len(card_data)}")
        if matches_found > 0:
            print(f"Average similarity score: {avg_similarity:.2f}")

        print(f"\nTo improve results:")
        print(f"1. Run multiple times to build up card database")
        print(f"2. Ensure good lighting and minimal glare in photos")
        print(f"3. Cards should be clearly visible and unobstructed")
    else:
        print("No cards processed successfully")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())