"""
03_enrichment_visualization.py

Assembles the four functional enrichment sub-panels into a single 
publication-ready vector PDF. Dynamically calculates dimensions and applies 
specific vertical shifts to eliminate inherent plotting margins.
"""

import warnings
from pathlib import Path

import fitz

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parents[2]
FIG_DIR = BASE_DIR / "outputs" / "figures"
OUT_FILE = FIG_DIR / "Fig4_Functional_Enrichment.pdf"

WIDTH_PT = 8.5 * 72

PANEL_FILES = {
    "A": FIG_DIR / "Fig4A.pdf",
    "B": FIG_DIR / "Fig4B.pdf",
    "C": FIG_DIR / "Fig4C.pdf",
    "D": FIG_DIR / "Fig4D.pdf"
}

def main():
    print("[*] Initiating Figure 4 Composite Assembly...")
    
    missing = [path.name for path in PANEL_FILES.values() if not path.exists()]
    if missing:
        print(f"[ERROR] Missing required panels: {', '.join(missing)}")
        return

    target_w = WIDTH_PT / 2.0
    
    panel_docs = {}
    panel_dims = {}

    for letter, path in PANEL_FILES.items():
        doc = fitz.open(path)
        panel_docs[letter] = doc
        src_rect = doc[0].rect
        
        scale = target_w / src_rect.width
        target_h = src_rect.height * scale
        
        panel_dims[letter] = {'w': target_w, 'h': target_h}

    row1_y = 0
    rect_A = fitz.Rect(0, row1_y, target_w, row1_y + panel_dims["A"]['h'])
    rect_B = fitz.Rect(target_w, row1_y, WIDTH_PT, row1_y + panel_dims["B"]['h'])
    
    VERTICAL_TRIM = -20 
    
    row2_y = row1_y + max(panel_dims["A"]['h'], panel_dims["B"]['h']) + VERTICAL_TRIM
    
    rect_C = fitz.Rect(0, row2_y, target_w, row2_y + panel_dims["C"]['h'])
    rect_D = fitz.Rect(target_w, row2_y, WIDTH_PT, row2_y + panel_dims["D"]['h'])

    quadrants = {"A": rect_A, "B": rect_B, "C": rect_C, "D": rect_D}

    master_height = row2_y + max(panel_dims["C"]['h'], panel_dims["D"]['h'])

    print("    -> Initializing dynamic master canvas...")
    master_doc = fitz.open()
    page = master_doc.new_page(width=WIDTH_PT, height=master_height)

    font_name = "helv"
    page.insert_font(fontname=font_name, fontbuffer=fitz.Font("helv").buffer)

    BOTTOM_LABEL_Y_OFFSET = 6 
    
    for letter, rect in quadrants.items():
        print(f"    -> Rendering Panel {letter}...")
        
        src_doc = panel_docs[letter]
        page.show_pdf_page(rect, src_doc, 0)
        
        label_x = rect.x0 + 15
        
        if letter in ["A", "B"]:
            label_y = rect.y0 + 30
        else:
            label_y = rect.y0 + BOTTOM_LABEL_Y_OFFSET 
        
        page.insert_text(
            (label_x, label_y), letter, fontsize=24, fontname=font_name, color=(0, 0, 0)
        )
        src_doc.close()

    master_doc.save(OUT_FILE)
    master_doc.close()
    
    print(f"[*] SUCCESS! Composite Figure 4 saved to: {OUT_FILE.name}")

if __name__ == "__main__":
    main()