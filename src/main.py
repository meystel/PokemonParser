#!/usr/bin/env python3
"""
Pokemon Card Visual Scanner - Alternative Approach
Uses image analysis and pattern matching instead of pure OCR
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
from PIL import Image, ImageDraw
import cv2
import numpy as np
import re


class PokemonVisualScanner:
    def __init__(self, offline_mode: bool = False):
        self.offline_mode = offline_mode
        self.api_base_url = "https://api.pokemontcg.io/v2/cards"
        self.last_api_call = 0
        self.api_delay = 0.1

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

    def analyze_card_colors(self, card_image: Image.Image) -> Dict[str, str]:
        """Analyze card colors to determine type and other characteristics."""
        # Convert to numpy array for analysis
        img_array = np.array(card_image)

        # Sample the border area (cards usually have colored borders)
        height, width = img_array.shape[:2]
        border_width = min(width, height) // 20

        # Sample top, bottom, left, right borders
        top_border = img_array[:border_width, :, :]
        bottom_border = img_array[-border_width:, :, :]
        left_border = img_array[:, :border_width, :]
        right_border = img_array[:, -border_width:, :]

        # Combine all border pixels
        border_pixels = np.concatenate([
            top_border.reshape(-1, 3),
            bottom_border.reshape(-1, 3),
            left_border.reshape(-1, 3),
            right_border.reshape(-1, 3)
        ])

        # Calculate dominant colors
        unique_colors, counts = np.unique(border_pixels, axis=0, return_counts=True)

        # Sort by frequency
        sorted_indices = np.argsort(counts)[::-1]
        dominant_colors = unique_colors[sorted_indices][:5]  # Top 5 colors

        # Classify card type based on dominant colors
        card_type = self.classify_card_type_by_color(dominant_colors)

        return {
            'dominant_colors': [color.tolist() for color in dominant_colors],
            'estimated_type': card_type
        }

    def classify_card_type_by_color(self, colors: np.ndarray) -> str:
        """Classify Pokemon card type based on border/dominant colors."""
        # Common Pokemon card type colors (approximate RGB values)
        type_colors = {
            'fire': ([255, 100, 100], [200, 50, 50]),  # Red
            'water': ([100, 150, 255], [50, 100, 200]),  # Blue
            'grass': ([100, 200, 100], [50, 150, 50]),  # Green
            'electric': ([255, 255, 100], [200, 200, 50]),  # Yellow
            'psychic': ([200, 100, 255], [150, 50, 200]),  # Purple
            'fighting': ([200, 150, 100], [150, 100, 50]),  # Brown
            'darkness': ([100, 100, 100], [50, 50, 50]),  # Dark gray
            'metal': ([200, 200, 200], [150, 150, 150]),  # Light gray
            'fairy': ([255, 200, 255], [200, 150, 200]),  # Pink
            'dragon': ([255, 200, 100], [200, 150, 50]),  # Gold
            'colorless': ([240, 240, 240], [200, 200, 200])  # Near white
        }

        best_match = 'unknown'
        best_score = float('inf')

        for color in colors[:3]:  # Check top 3 dominant colors
            for card_type, (target_high, target_low) in type_colors.items():
                # Calculate distance to both high and low variants
                dist_high = np.linalg.norm(color - np.array(target_high))
                dist_low = np.linalg.norm(color - np.array(target_low))
                dist = min(dist_high, dist_low)

                if dist < best_score:
                    best_score = dist
                    best_match = card_type

        return best_match if best_score < 100 else 'unknown'

    def detect_card_features(self, card_image: Image.Image) -> Dict[str, any]:
        """Detect visual features of the card."""
        # Convert to OpenCV format
        cv_image = cv2.cvtColor(np.array(card_image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        features = {}

        # Detect if card has holographic/shiny effects
        features['has_holo_effect'] = self.detect_holographic_effect(gray)

        # Detect text regions (even if we can't read them well)
        features['text_regions'] = self.detect_text_regions(gray)

        # Calculate image hash for potential matching
        features['image_hash'] = self.calculate_image_hash(card_image)

        # Analyze card layout
        features['layout_type'] = self.analyze_card_layout(gray)

        return features

    def detect_holographic_effect(self, gray_image: np.ndarray) -> bool:
        """Detect if the card has holographic/foil effects."""
        # Holographic cards often have high variance in pixel intensity
        # due to reflections and rainbow effects

        # Calculate local variance
        kernel = np.ones((5, 5), np.float32) / 25
        mean = cv2.filter2D(gray_image.astype(np.float32), -1, kernel)
        variance = cv2.filter2D((gray_image.astype(np.float32) - mean) ** 2, -1, kernel)

        # High variance regions suggest holographic effects
        high_variance_threshold = np.percentile(variance, 90)
        high_variance_pixels = np.sum(variance > high_variance_threshold)
        total_pixels = variance.size

        variance_ratio = high_variance_pixels / total_pixels
        return variance_ratio > 0.05  # 5% of pixels have high variance

    def detect_text_regions(self, gray_image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect regions that likely contain text."""
        # Use MSER (Maximally Stable Extremal Regions) to find text-like regions
        mser = cv2.MSER_create()
        regions, _ = mser.detectRegions(gray_image)

        text_regions = []
        for region in regions:
            # Calculate bounding box
            x, y, w, h = cv2.boundingRect(region.reshape(-1, 1, 2))

            # Filter regions that are likely to be text
            aspect_ratio = w / h if h > 0 else 0
            area = w * h

            # Text regions usually have reasonable aspect ratios and sizes
            if 0.1 < aspect_ratio < 10 and 100 < area < 5000:
                text_regions.append((x, y, w, h))

        return text_regions

    def calculate_image_hash(self, image: Image.Image) -> str:
        """Calculate a perceptual hash of the card image."""
        # Resize to small size and convert to grayscale
        small_image = image.resize((8, 8), Image.LANCZOS).convert('L')
        pixels = list(small_image.getdata())

        # Calculate average
        avg = sum(pixels) / len(pixels)

        # Create hash based on whether each pixel is above/below average
        hash_bits = ''.join(['1' if pixel > avg else '0' for pixel in pixels])

        # Convert to hex
        hash_hex = hex(int(hash_bits, 2))[2:].zfill(16)
        return hash_hex

    def analyze_card_layout(self, gray_image: np.ndarray) -> str:
        """Analyze the overall layout of the card."""
        height, width = gray_image.shape

        # Divide into regions and analyze
        top_region = gray_image[:height // 3, :]
        middle_region = gray_image[height // 3:2 * height // 3, :]
        bottom_region = gray_image[2 * height // 3:, :]

        # Calculate average intensity for each region
        top_avg = np.mean(top_region)
        middle_avg = np.mean(middle_region)
        bottom_avg = np.mean(bottom_region)

        # Classify based on intensity distribution
        if middle_avg < top_avg and middle_avg < bottom_avg:
            return 'pokemon_card'  # Typical Pokemon card has darker middle (artwork)
        elif top_avg > middle_avg > bottom_avg:
            return 'trainer_card'  # Might be a trainer card
        else:
            return 'unknown_layout'

    def search_card_by_features(self, features: Dict) -> Optional[Dict]:
        """Search for card using visual features instead of OCR text."""
        if self.offline_mode:
            return None

        try:
            # For now, we'll do a broad search and let the user manually verify
            # In a real implementation, you might:
            # 1. Use image similarity matching
            # 2. Build a database of card hashes
            # 3. Use machine learning for card recognition

            # Search for cards of the detected type
            estimated_type = features.get('estimated_type', '')
            if estimated_type and estimated_type != 'unknown':
                params = {
                    'q': f'types:{estimated_type}',
                    'select': 'name,number,set,types,cardmarket',
                    'pageSize': 5  # Just get a few examples
                }

                response = requests.get(self.api_base_url, params=params, timeout=10)
                response.raise_for_status()

                data = response.json()
                if data.get('data'):
                    # Return the first match as an example
                    # In practice, you'd want better matching logic
                    return data['data'][0]

        except Exception as e:
            print(f"Error searching by features: {e}")

        return None

    def process_cards(self, image_path: str, output_dir: str) -> List[Dict]:
        """Process all cards using visual analysis instead of OCR."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        print("Splitting image into individual cards...")
        cards = self.split_image_3x3(image_path)

        if not cards:
            print("Failed to split image")
            return []

        results = []

        for i, card_image in enumerate(cards, 1):
            print(f"Analyzing card {i}/9...")

            # Save cropped card image
            card_filename = f"card_{i:02d}.jpg"
            card_path = os.path.join(output_dir, card_filename)
            card_image.save(card_path, 'JPEG', quality=95)

            # Analyze visual features instead of OCR
            color_analysis = self.analyze_card_colors(card_image)
            visual_features = self.detect_card_features(card_image)

            # Combine all analysis
            card_info = {
                'card_number': i,
                'image_file': card_filename,
                'estimated_type': color_analysis['estimated_type'],
                'has_holo_effect': visual_features['has_holo_effect'],
                'layout_type': visual_features['layout_type'],
                'image_hash': visual_features['image_hash'],
                'text_regions_count': len(visual_features['text_regions']),
                'dominant_colors': str(color_analysis['dominant_colors']),
                'name': f'Unknown Card {i}',  # Placeholder
                'number': '',
                'set': '',
                'rarity': 'holo' if visual_features['has_holo_effect'] else 'unknown'
            }

            # Try to get additional info if online
            if not self.offline_mode:
                print(f"  Searching for {color_analysis['estimated_type']} type cards...")
                api_result = self.search_card_by_features({
                    'estimated_type': color_analysis['estimated_type'],
                    'has_holo_effect': visual_features['has_holo_effect']
                })

                if api_result:
                    card_info.update({
                        'api_suggestion_name': api_result.get('name', ''),
                        'api_suggestion_number': api_result.get('number', ''),
                        'api_suggestion_set': api_result.get('set', {}).get('name', ''),
                        'market_price': 0.0,  # Would need specific card lookup
                    })

            results.append(card_info)

        return results

    def save_to_csv(self, card_data: List[Dict], output_path: str):
        """Save card analysis to CSV file."""
        if not card_data:
            print("No card data to save")
            return

        headers = [
            'card_number', 'image_file', 'estimated_type', 'has_holo_effect',
            'layout_type', 'text_regions_count', 'rarity', 'image_hash'
        ]

        if not self.offline_mode:
            headers.extend(['api_suggestion_name', 'api_suggestion_number', 'api_suggestion_set'])

        try:
            with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=headers)
                writer.writeheader()

                for card in card_data:
                    row = {header: card.get(header, '') for header in headers}
                    writer.writerow(row)

            print(f"CSV saved to: {output_path}")

        except Exception as e:
            print(f"Error saving CSV: {e}")


def main():
    parser = argparse.ArgumentParser(description='Pokemon Card Visual Scanner')
    parser.add_argument('input_image', help='Path to input image (3x3 grid of Pokemon cards)')
    parser.add_argument('output_dir', help='Output directory for cropped images and CSV')
    parser.add_argument('--offline', action='store_true',
                        help='Run in offline mode (no API calls)')

    args = parser.parse_args()

    if not os.path.exists(args.input_image):
        print(f"Error: Input image '{args.input_image}' not found")
        return 1

    scanner = PokemonVisualScanner(offline_mode=args.offline)

    print(f"Processing image: {args.input_image}")
    print(f"Output directory: {args.output_dir}")
    print(f"Mode: {'Offline' if args.offline else 'Online'}")
    print("Note: This version uses visual analysis instead of OCR")

    card_data = scanner.process_cards(args.input_image, args.output_dir)

    if card_data:
        csv_path = os.path.join(args.output_dir, 'pokemon_cards_visual.csv')
        scanner.save_to_csv(card_data, csv_path)

        print(f"\nProcessing complete!")
        print(f"Analyzed {len(card_data)} cards")
        print(f"Images saved to: {args.output_dir}")
        print(f"Analysis saved to: {csv_path}")

        # Show summary of detected types
        type_counts = {}
        holo_count = 0
        for card in card_data:
            card_type = card.get('estimated_type', 'unknown')
            type_counts[card_type] = type_counts.get(card_type, 0) + 1
            if card.get('has_holo_effect'):
                holo_count += 1

        print(f"\nDetected card types: {dict(type_counts)}")
        print(f"Holographic cards detected: {holo_count}")
        print(f"\nNote: Card names need manual verification or better image recognition")
    else:
        print("No cards processed successfully")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())