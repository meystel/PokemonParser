# 🃏 Pokemon Card Scanner

> **Automatically analyze Pokemon card collections with computer vision and OCR**

Transform your messy pile of Pokemon cards into organized data! This tool takes a photo of a 3x3 grid of cards and extracts valuable information about each one.

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-active-success.svg)]()

## 🚀 Features

### Image Processing
- **Smart Grid Splitting**: Automatically divides 3x3 card grids into individual card images
- **Advanced OCR**: Multiple preprocessing approaches for better text recognition
- **Visual Analysis**: Color-based type detection and holographic effect recognition

### Data Extraction
- **Card Information**: Extracts names, numbers, sets, and rarity
- **Market Pricing**: Integrates with Pokemon TCG API for current market values
- **Export Options**: Clean CSV output with all card data

### Modes
- **🌐 Online Mode**: Full API integration for pricing and card verification
- **📱 Offline Mode**: Works without internet for basic analysis

## 📸 How It Works

```
[3x3 Grid Photo] → [9 Individual Cards] → [OCR Analysis] → [CSV + Images]
```

1. **Take a photo** of 9 Pokemon cards arranged in a 3x3 grid
2. **Run the scanner** - it splits and analyzes each card
3. **Get results** - cropped images + CSV with all the data

## 🛠 Installation

### Prerequisites
```bash
# Install Tesseract OCR
brew install tesseract  # macOS
# sudo apt-get install tesseract-ocr  # Ubuntu
```

### Python Dependencies
```bash
pip install -r requirements.txt
```

Required packages:
- `requests` - Pokemon TCG API calls
- `Pillow` - Image processing
- `pytesseract` - OCR text extraction
- `opencv-python` - Advanced image preprocessing
- `numpy` - Numerical operations

## 🎯 Usage

### Basic Usage
```bash
# Online mode (with pricing)
python src/main.py cards_photo.jpg output_folder

# Offline mode (no API calls)
python src/main.py cards_photo.jpg output_folder --offline
```

### Example
```bash
python src/main.py my_pokemon_cards.jpg results/
```

**Output:**
```
results/
├── card_01.jpg
├── card_02.jpg
├── ...
├── card_09.jpg
└── pokemon_cards.csv
```

## 📊 CSV Output

| Column | Description |
|--------|-------------|
| `card_number` | Position in grid (1-9) |
| `image_file` | Filename of cropped card |
| `name` | Pokemon/card name |
| `number` | Card number (e.g. "123/456") |
| `set` | Set abbreviation |
| `rarity` | Card rarity |
| `market_price` | Current market value |
| `raw_text` | Full OCR output |

## 🎨 Visual Analysis Version

For cards with difficult-to-read text, try the visual analysis version:

```python
# Uses color analysis instead of OCR
from pokemon_visual_scanner import PokemonVisualScanner
```

Features:
- **Type Detection**: Fire 🔥, Water 💧, Grass 🌱, etc. based on card colors
- **Holographic Detection**: Identifies foil and special cards
- **Layout Analysis**: Distinguishes Pokemon vs Trainer cards

## 🤖 API Integration

Integrates with [Pokemon TCG API](https://pokemontcg.io/) for:
- Card verification
- Current market prices
- Set information
- Rarity confirmation

Rate limited to respect API guidelines.

## 📋 Requirements

- **Python 3.8+**
- **Tesseract OCR** installed on system
- **Good lighting** for photos (avoid shadows and glare)
- **Clear card arrangement** (3x3 grid with some spacing)

## 🔧 Troubleshooting

### OCR Not Working?
- Ensure good lighting when taking photos
- Try the visual analysis version for stylized cards
- Check that Tesseract is properly installed

### API Issues?
- Use `--offline` mode if API is down
- Check your internet connection
- API has rate limits - the tool includes delays

### Poor Results?
- Make sure cards are flat and well-lit
- Avoid shadows and reflections
- Try individual card photos instead of grids

## 🚧 Known Limitations

- **OCR Accuracy**: Stylized fonts and foil effects can confuse text recognition
- **Grid Detection**: Works best with evenly spaced 3x3 arrangements
- **Lighting Sensitive**: Poor lighting significantly affects results
- **API Dependent**: Pricing requires internet connection

## 🛣 Roadmap

- [ ] Support for other grid sizes (2x2, 4x4, etc.)
- [ ] Machine learning for better card recognition
- [ ] Web interface for easier use
- [ ] Support for other TCGs (Magic, Yu-Gi-Oh, etc.)
- [ ] Mobile app version

## 🤝 Contributing

Found a bug? Have an idea? Contributions welcome!

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Pokemon TCG API](https://pokemontcg.io/) for card data
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) for text recognition
- OpenCV community for image processing tools

---

**⚡ Pro Tip**: For best results, take photos in natural lighting with cards laid flat on a dark background!

---

<p align="center">
  <strong>Happy Card Scanning! 🎮✨</strong>
</p>