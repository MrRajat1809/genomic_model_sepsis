"""
02_deg_visualization.py

Assembles the three DEG sub-panels (Heatmap, Volcano, Correlation Matrix) 
into a single publication-ready vector PDF. 
Implements a 1x3 horizontal layout by anchoring a uniform target height 
and dynamically scaling widths to preserve exact vector aspect ratios.
"""

import warnings
from pathlib import Path

import fitz  # PyMuPDF

warnings.filterwarnings("ignore")

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = Path(__file__).resolve().parents[2]
FIG_DIR = BASE_DIR / "outputs" / "figures"
OUT_FILE = FIG_DIR / "Fig3_DEG_Composite.pdf"

PANEL_FILES = {
    "A": FIG_DIR / "Fig3A.pdf",
    "B": FIG_DIR / "Fig3B.pdf",
    "C": FIG_DIR / "Fig3C.pdf"
}

# Anchor Height: 7.5 inches (1 inch = 72 points)
# This provides a large, high-resolution horizontal canvas suitable for a full-page width layout.
TARGET_HEIGHT_PT = 7.5 * 72

# Horizontal padding between panels (in points)
PADDING_PT = 20

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    print("[*] Initiating Figure 3 Horizontal Composite Assembly...")

    # 1. Validate inputs
    missing = [path.name for path in PANEL_FILES.values() if not path.exists()]
    if missing:
        print(f"[ERROR] Missing required panels: {', '.join(missing)}")
        return

    # ---------------------------------------------------------
    # 2. DYNAMIC GEOMETRY CALCULATION (Height-Anchored)
    # ---------------------------------------------------------
    panel_docs = {}
    panel_dims = {}
    
    total_width = 0

    for letter, path in PANEL_FILES.items():
        doc = fitz.open(path)
        panel_docs[letter] = doc
        src_rect = doc[0].rect
        
        # Calculate scale factor to force uniform height while preserving aspect ratio
        scale = TARGET_HEIGHT_PT / src_rect.height
        target_w = src_rect.width * scale
        
        panel_dims[letter] = {'w': target_w, 'h': TARGET_HEIGHT_PT}
        
        # Accumulate total width
        total_width += target_w

    # Add padding to the total canvas width (2 gaps for 3 panels)
    total_width += (PADDING_PT * 2)

    # ---------------------------------------------------------
    # 3. BUILD EXACT BOUNDING BOXES (Left-to-Right)
    # ---------------------------------------------------------
    rects = {}
    current_x = 0
    
    for letter in ["A", "B", "C"]:
        w = panel_dims[letter]['w']
        rects[letter] = fitz.Rect(current_x, 0, current_x + w, TARGET_HEIGHT_PT)
        current_x += (w + PADDING_PT)

    # ---------------------------------------------------------
    # 4. RENDER AND ASSEMBLE
    # ---------------------------------------------------------
    print(f"    -> Initializing master canvas (Width: {total_width/72:.1f}in, Height: {TARGET_HEIGHT_PT/72:.1f}in)...")
    master_doc = fitz.open()
    page = master_doc.new_page(width=total_width, height=TARGET_HEIGHT_PT)

    # Load standard Helvetica font for journal-compliant panel labels
    font_name = "helv"
    page.insert_font(fontname=font_name, fontbuffer=fitz.Font("helv").buffer)

    for letter in ["A", "B", "C"]:
        print(f"    -> Rendering Panel {letter}...")
        
        src_doc = panel_docs[letter]
        rect = rects[letter]
        
        # Inject the PDF page vector data
        page.show_pdf_page(rect, src_doc, 0)
        
        # Label placement (Top-Left of each panel's specific bounding box)
        label_x = rect.x0 + 10
        label_y = rect.y0 + 25
        
        page.insert_text(
            (label_x, label_y), letter, fontsize=28, fontname=font_name, color=(0, 0, 0)
        )
        
        src_doc.close()

    # ---------------------------------------------------------
    # 5. EXPORT
    # ---------------------------------------------------------
    master_doc.save(OUT_FILE)
    master_doc.close()
    
    print(f"[*] SUCCESS! 1x3 Composite Figure 3 saved to: {OUT_FILE.name}")

if __name__ == "__main__":
    main()