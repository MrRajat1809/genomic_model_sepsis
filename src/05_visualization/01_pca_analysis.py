"""
01_pca_analysis.py

Assembles the two PCA diagnostic panels (Fig 2A and Fig 2B) into a single
publication-ready vector PDF. Stacks the sub-panels vertically, preserves 
strict vector geometry, and injects a centered statistical validation footer.
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
OUT_FILE = FIG_DIR / "Fig2_PCA_Analysis.pdf"

# Standard journal page width (~8.5 inches). 1 inch = 72 points.
WIDTH_PT = 8.5 * 72

PANEL_FILES = {
    "A": FIG_DIR / "Fig2A.pdf",
    "B": FIG_DIR / "Fig2B.pdf"
}

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    print("[*] Initiating Figure 2 Composite Assembly...")

    missing = [path.name for path in PANEL_FILES.values() if not path.exists()]
    if missing:
        print(f"[ERROR] Missing required panels: {', '.join(missing)}")
        return

    # ---------------------------------------------------------
    # 1. DYNAMIC GEOMETRY CALCULATION
    # ---------------------------------------------------------
    panel_docs = {}
    panel_dims = {}

    for letter, path in PANEL_FILES.items():
        doc = fitz.open(path)
        panel_docs[letter] = doc
        src_rect = doc[0].rect

        # Scale proportionally to fit the target master canvas width
        scale = WIDTH_PT / src_rect.width
        target_h = src_rect.height * scale

        panel_dims[letter] = {'w': WIDTH_PT, 'h': target_h}

    # ---------------------------------------------------------
    # 2. BUILD EXACT BOUNDING BOXES
    # ---------------------------------------------------------
    row1_y = 0
    rect_A = fitz.Rect(0, row1_y, WIDTH_PT, row1_y + panel_dims["A"]['h'])

    VERTICAL_TRIM = -4 

    row2_y = row1_y + panel_dims["A"]['h'] + VERTICAL_TRIM
    rect_B = fitz.Rect(0, row2_y, WIDTH_PT, row2_y + panel_dims["B"]['h'])

    STATS_HEIGHT = 50
    master_height = row2_y + panel_dims["B"]['h'] + STATS_HEIGHT

    # ---------------------------------------------------------
    # 3. RENDER PANELS
    # ---------------------------------------------------------
    print("    -> Initializing master vector canvas...")
    master_doc = fitz.open()
    page = master_doc.new_page(width=WIDTH_PT, height=master_height)

    print("    -> Rendering Panel A (Top)...")
    page.show_pdf_page(rect_A, panel_docs["A"], 0)
    panel_docs["A"].close()

    print("    -> Rendering Panel B (Bottom)...")
    page.show_pdf_page(rect_B, panel_docs["B"], 0)
    panel_docs["B"].close()

    # ---------------------------------------------------------
    # 4. INJECT STATISTICAL VALIDATION FOOTER
    # ---------------------------------------------------------
    print("    -> Injecting ANOVA statistical summary footer...")
    
    font_name_reg = "helv"
    font_name_bold = "helvb"
    
    # Load fonts into the page
    page.insert_font(fontname=font_name_reg, fontbuffer=fitz.Font("helv").buffer)
    page.insert_font(fontname=font_name_bold, fontbuffer=fitz.Font("helv", is_bold=True).buffer)
    
    # Load font objects to calculate exact string widths
    font_reg = fitz.Font("helv")
    font_bold = fitz.Font("helv", is_bold=True)

    # Draw separator line
    line_y = row2_y + panel_dims["B"]['h'] + 10
    page.draw_line(fitz.Point(40, line_y), fitz.Point(WIDTH_PT - 40, line_y), color=(0.7, 0.7, 0.7), width=0.5)

    TITLE_SIZE = 9
    TEXT_SIZE = 8

    # Text content to inject
    str_title = "Statistical Validation (ANOVA)"
    str_line1 = "- Technical Batch (Cohort): PC1-PC3 p = 1.000 (Variance Neutralized)"
    str_line2 = "- Biological Signal (Mortality): PC3 F = 11.88, p < 0.001 (Signal Preserved)"

    # Calculate X coordinates to dynamically center each string based on text length
    x_title = (WIDTH_PT - font_bold.text_length(str_title, fontsize=TITLE_SIZE)) / 2.0
    x_line1 = (WIDTH_PT - font_reg.text_length(str_line1, fontsize=TEXT_SIZE)) / 2.0
    x_line2 = (WIDTH_PT - font_reg.text_length(str_line2, fontsize=TEXT_SIZE)) / 2.0

    # Insert text using precise baseline coordinates
    page.insert_text((x_title, line_y + 15), str_title, fontname=font_name_bold, fontsize=TITLE_SIZE, color=(0, 0, 0))
    page.insert_text((x_line1, line_y + 26), str_line1, fontname=font_name_reg, fontsize=TEXT_SIZE, color=(0.2, 0.2, 0.2))
    page.insert_text((x_line2, line_y + 37), str_line2, fontname=font_name_reg, fontsize=TEXT_SIZE, color=(0.2, 0.2, 0.2))

    # ---------------------------------------------------------
    # 5. EXPORT
    # ---------------------------------------------------------
    master_doc.save(OUT_FILE)
    master_doc.close()

    print(f"[*] SUCCESS! Composite Figure 2 saved to: {OUT_FILE.name}")

if __name__ == "__main__":
    main()