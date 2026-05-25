"""
06_statistics_assembly.py

Assembles the Final Validation Sub-panels into a single composite figure.
Implements a block-justified 2-row layout scaled for a standard journal page width.
Utilizes a manual Y-shift matrix to align the visual data areas of adjacent panels 
despite differing internal margins.
"""

import warnings
from pathlib import Path

import fitz

warnings.filterwarnings("ignore")

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = Path(__file__).resolve().parents[2]
FIG_DIR = BASE_DIR / "outputs" / "figures"
OUT_FILE = FIG_DIR / "Fig7_Validation_Composite.pdf"

panel_A_path = FIG_DIR / "Fig7A.pdf"
if not panel_A_path.exists() and (FIG_DIR / "Fig_Combined_Validation_ROC.pdf").exists():
    panel_A_path = FIG_DIR / "Fig_Combined_Validation_ROC.pdf"

PANEL_FILES = {
    "7A": panel_A_path,          
    "7B": FIG_DIR / "Fig7B.pdf", 
    "7C": FIG_DIR / "Fig7C.pdf", 
    "7D": FIG_DIR / "Fig7D.pdf"  
}

TARGET_WIDTH_PT = 8.0 * 72  

PADDING_X = 20  
PADDING_Y = 30  
MARGIN = 20     

LABEL_FONT = "hebo"
LABEL_SIZE = 18       
LABEL_OFFSET_X = 5    
LABEL_OFFSET_Y = 10   

# ==========================================
# MANUAL ALIGNMENT MATRIX
# ==========================================
# Shifts specific panels down (positive) or up (negative) in points 
# to align the plotted data areas across the row.
PANEL_Y_SHIFTS = {
    "7A": 0,
    "7B": 0,
    "7C": 18,  
    "7D": 0
}

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    print("Initiating Figure 7 Assembly...")

    missing = [path.name for path in PANEL_FILES.values() if not path.exists()]
    if missing:
        print(f"Error: Missing required panels: {', '.join(missing)}")
        return

    # ---------------------------------------------------------
    # 1. LOAD DOCUMENTS & EXTRACT ASPECT RATIOS
    # ---------------------------------------------------------
    docs = {key: fitz.open(path) for key, path in PANEL_FILES.items()}
    ars = {} 
    
    for key, doc in docs.items():
        rect = doc[0].rect
        ars[key] = rect.width / rect.height

    # ---------------------------------------------------------
    # 2. DYNAMIC BLOCK-JUSTIFICATION MATH
    # ---------------------------------------------------------
    usable_width = TARGET_WIDTH_PT - (MARGIN * 2) - PADDING_X
    
    row1_height = usable_width / (ars["7A"] + ars["7B"])
    row2_height = usable_width / (ars["7C"] + ars["7D"])
    
    dims = {
        "7A": {'w': row1_height * ars["7A"], 'h': row1_height},
        "7B": {'w': row1_height * ars["7B"], 'h': row1_height},
        "7C": {'w': row2_height * ars["7C"], 'h': row2_height},
        "7D": {'w': row2_height * ars["7D"], 'h': row2_height}
    }

    max_shift = max(PANEL_Y_SHIFTS.values())
    master_height = MARGIN + row1_height + PADDING_Y + row2_height + MARGIN + max_shift
    print(f"Master Canvas initialized: {TARGET_WIDTH_PT/72:.1f} x {master_height/72:.1f} inches")

    # ---------------------------------------------------------
    # 3. DEFINE EXACT BOUNDING BOXES (With Shifts Applied)
    # ---------------------------------------------------------
    r1_y = MARGIN
    rect_7A = fitz.Rect(MARGIN, r1_y + PANEL_Y_SHIFTS["7A"], 
                        MARGIN + dims["7A"]['w'], r1_y + dims["7A"]['h'] + PANEL_Y_SHIFTS["7A"])
    
    x_7B = MARGIN + dims["7A"]['w'] + PADDING_X
    rect_7B = fitz.Rect(x_7B, r1_y + PANEL_Y_SHIFTS["7B"], 
                        x_7B + dims["7B"]['w'], r1_y + dims["7B"]['h'] + PANEL_Y_SHIFTS["7B"])

    r2_y = r1_y + row1_height + PADDING_Y
    
    rect_7C = fitz.Rect(MARGIN, r2_y + PANEL_Y_SHIFTS["7C"], 
                        MARGIN + dims["7C"]['w'], r2_y + dims["7C"]['h'] + PANEL_Y_SHIFTS["7C"])
    
    x_7D = MARGIN + dims["7C"]['w'] + PADDING_X
    rect_7D = fitz.Rect(x_7D, r2_y + PANEL_Y_SHIFTS["7D"], 
                        x_7D + dims["7D"]['w'], r2_y + dims["7D"]['h'] + PANEL_Y_SHIFTS["7D"])

    # ---------------------------------------------------------
    # 4. RENDER PANELS TO MASTER CANVAS
    # ---------------------------------------------------------
    master_doc = fitz.open()
    page = master_doc.new_page(width=TARGET_WIDTH_PT, height=master_height)
    
    page.insert_font(fontname=LABEL_FONT, fontbuffer=fitz.Font(LABEL_FONT).buffer)

    print("Stitching Row 1 (Panels A, B, C)...")
    page.show_pdf_page(rect_7A, docs["7A"], 0)
    page.show_pdf_page(rect_7B, docs["7B"], 0)
    
    print("Stitching Row 2 (Panels D, E)...")
    page.show_pdf_page(rect_7C, docs["7C"], 0)
    page.show_pdf_page(rect_7D, docs["7D"], 0)

    # ---------------------------------------------------------
    # 5. INJECT VECTOR TYPOGRAPHY (PANEL LABELS)
    # ---------------------------------------------------------
    labels = {
        "A": (rect_7A.x0 + LABEL_OFFSET_X, rect_7A.y0 + LABEL_OFFSET_Y),
        "B": (rect_7A.x0 + (dims["7A"]['w'] / 2) + LABEL_OFFSET_X, rect_7A.y0 + LABEL_OFFSET_Y),
        "C": (rect_7B.x0 + LABEL_OFFSET_X, rect_7B.y0 + LABEL_OFFSET_Y),
        "D": (rect_7C.x0 + LABEL_OFFSET_X, rect_7C.y0 + LABEL_OFFSET_Y),
        "E": (rect_7D.x0 + LABEL_OFFSET_X, rect_7D.y0 + LABEL_OFFSET_Y)
    }

    for letter, coord in labels.items():
        page.insert_text(
            coord, letter, fontsize=LABEL_SIZE, fontname=LABEL_FONT, color=(0, 0, 0)
        )

    # ---------------------------------------------------------
    # 6. CLEANUP & EXPORT
    # ---------------------------------------------------------
    for doc in docs.values():
        doc.close()

    master_doc.save(OUT_FILE)
    master_doc.close()
    
    print(f"Assembly complete. Composite saved to: {OUT_FILE.name}")

if __name__ == "__main__":
    main()