import os
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

# 1. Define output directory
OUTPUT_DIR = BASE_DIR / "data" / "processed" / "camouflaged_object_image"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# 2. Define image parameters
IMG_SIZE = 1024
BG_COLOR = (255, 255, 255)  # Pure white
TEXT_COLOR = (0, 0, 0)      # Pure black
PADDING = 5                 # Padding to prevent text from touching edges

# 3. Define data list (id, Super, Sub, Text, Filename)
# The symbol smiley face ☺ is not drawn via code
data_list = [
    (1, "the character T", "001_textual_elements_single_character_t.png", "T"),
    (2, "the character X", "002_textual_elements_single_character_x.png", "X"),
    (3, "the character S", "003_textual_elements_single_character_s.png", "S"),
    (4, "the character 2", "004_textual_elements_single_character_2.png", "2"),
    (5, "the character A", "005_textual_elements_single_character_a.png", "A"),
    (6, "the character P", "006_textual_elements_single_character_p.png", "P"),
    (7, "the character 8", "007_textual_elements_single_character_8.png", "8"),
    (8, "the ampersand symbol (&)", "008_textual_elements_single_character_ampersand.png", "&"),
    (9, "the word AI", "009_textual_elements_short_word_ai.png", "AI"),
    (10, "the word CAT", "010_textual_elements_short_word_cat.png", "CAT"),
    (11, "the word DOG", "011_textual_elements_short_word_dog.png", "DOG"),
    (12, "the word SKY", "012_textual_elements_short_word_sky.png", "SKY"),
    (13, "the word DREAM", "013_textual_elements_short_word_dream.png", "DREAM"),
    (14, "the word MAGIC", "014_textual_elements_short_word_magic.png", "MAGIC"),
    (15, "the word NATURE", "015_textual_elements_short_word_nature.png", "NATURE"),
    (16, "the word FUTURE", "016_textual_elements_short_word_future.png", "FUTURE"),
    (17, "the word INTELLIGENCE", "017_textual_elements_long_word_intelligence.png", "INTELLIGENCE"),
    (18, "the word SUSTAINABILITY", "018_textual_elements_long_word_sustainability.png", "SUSTAINABILITY"),
    (19, "the word CRYPTOCURRENCY", "019_textual_elements_long_word_cryptocurrency.png", "CRYPTOCURRENCY"),
    (20, "the word CONGRATULATIONS", "020_textual_elements_long_word_congratulations.png", "CONGRATULATIONS"),
    (21, "the word REVOLUTIONARY", "021_textual_elements_long_word_revolutionary.png", "REVOLUTIONARY"),
    (22, "the word ARCHITECTURAL", "022_textual_elements_long_word_architectural.png", "ARCHITECTURAL"),
    (23, "the word PHOTOGRAPHY", "023_textual_elements_long_word_photography.png", "PHOTOGRAPHY"),
    (24, "the word PHILOSOPHY", "024_textual_elements_long_word_philosophy.png", "PHILOSOPHY"),
    (25, "the Chinese character 人", "025_textual_elements_non_latin_character_char_cn_ren.png", "人"),
    (26, "the Chinese character 中", "026_textual_elements_non_latin_character_char_cn_zhong.png", "中"),
    (27, "the Chinese character 好", "027_textual_elements_non_latin_character_char_cn_hao.png", "好"),
    (28, "the Chinese character 赢", "028_textual_elements_non_latin_character_char_cn_ying.png", "赢"),
    (29, "the Japanese Hiragana character あ", "029_textual_elements_non_latin_character_char_jp_a.png", "あ"),
    (30, "the Japanese Hiragana character の", "030_textual_elements_non_latin_character_char_jp_no.png", "の"),
    # (31, "the symbol ☺ (smiley face)", "031_textual_elements_non_latin_character_symbol_smiley.png", "☺"),
    (32, "the uppercase Greek letter Ω", "032_textual_elements_non_latin_character_symbol_omega.png", "Ω"),
]

def get_optimal_font(draw, text, max_width, max_height, font_path):
    """
    Binary search for the optimal font size so the text fills the canvas as much as possible.
    """
    min_size = 10
    max_size = 1200  # Slightly larger than image size, as upper bound
    optimal_font = None
    
    while min_size <= max_size:
        mid_size = (min_size + max_size) // 2
        try:
            font = ImageFont.truetype(font_path, mid_size)
        except OSError:
            # If loading fails, return default font (usually small, mainly for debugging)
            return ImageFont.load_default()

        # Get text bounding box (left, top, right, bottom)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        if text_w <= max_width and text_h <= max_height:
            optimal_font = font
            min_size = mid_size + 1
        else:
            max_size = mid_size - 1
            
    return optimal_font

def create_text_image(filename, text_content):
    # Create canvas
    img = Image.new('RGB', (IMG_SIZE, IMG_SIZE), color=BG_COLOR)
    draw = ImageDraw.Draw(img)
    
    # === Font selection strategy ===
    # To support Chinese, Japanese, and special symbols, we need appropriate font files.
    # Windows: "msyh.ttc" (Microsoft YaHei), "simhei.ttf" (SimHei), "arialuni.ttf" (Arial Unicode MS)
    # Mac: "Arial Unicode.ttf", "PingFang.ttc"
    # Linux: "NotoSansCJK-Regular.ttc"

    # A list of fallback fonts; the script will try them in order.
    font_candidates = [
        "arial.ttf",           # Preferred for English
        "msyh.ttc",            # Windows Chinese
        "simhei.ttf",          # Windows Chinese (SimHei)
        "PingFang.ttc",        # Mac Chinese
        "NotoSansCJK-Regular.ttc", # Linux Chinese
        "AppleGothic.ttf",     # Mac Korean/Japanese
        "msgothic.ttc",        # Windows Japanese
        "seguiemj.ttf"         # Windows Emoji (for smiley faces)
    ]

    # Special handling: if text contains Chinese, Japanese, or symbols, prefer Unicode fonts first
    is_special = any(ord(c) > 127 for c in text_content)
    if is_special:
        # Prioritize wide-character fonts
        font_candidates = ["msyh.ttc", "simhei.ttf", "msgothic.ttc", "arialuni.ttf"] + font_candidates

    selected_font_path = None
    
    # Try to find a font that exists on the system
    for font_name in font_candidates:
        try:
            # Try loading to check if it exists
            ImageFont.truetype(font_name, 40)
            selected_font_path = font_name
            break
        except OSError:
            continue
            
    if selected_font_path is None:
        print(f"Warning: No suitable font file found, will use default font for '{text_content}' (non-ASCII characters may not render)")
        # Default font logic differs slightly; omitted here for brevity. Ensure the system has the fonts listed above.
        selected_font_path = "arial.ttf"  # Fallback; will error if missing — specify an absolute path if needed

    # Auto-calculate maximum available area
    max_w = IMG_SIZE - 2 * PADDING
    max_h = IMG_SIZE - 2 * PADDING

    # Get the best-fit font object
    font = get_optimal_font(draw, text_content, max_w, max_h, selected_font_path)

    # Calculate text centering position
    bbox = draw.textbbox((0, 0), text_content, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    # Adjust vertical centering: Pillow's baseline may cause visual offset; use bbox center to align with image center
    # textbbox (0,0) is the top-left, but the draw xy typically refers to baseline or top-left.
    # Simple centering algorithm:
    x = (IMG_SIZE - text_w) / 2 - bbox[0]
    y = (IMG_SIZE - text_h) / 2 - bbox[1]

    # Draw text
    draw.text((x, y), text_content, font=font, fill=TEXT_COLOR)

    # Save
    save_path = os.path.join(OUTPUT_DIR, filename)
    img.save(save_path)
    print(f"Generated: {save_path} (content: {text_content})")

# === Main ===
if __name__ == "__main__":
    print("Starting text image generation...")

    for idx, desc, filename, content in data_list:
        try:
            create_text_image(filename, content)
        except Exception as e:
            print(f"Generation failed for ID {idx}: {e}")

    print("All images generated. Please check the output folder.")