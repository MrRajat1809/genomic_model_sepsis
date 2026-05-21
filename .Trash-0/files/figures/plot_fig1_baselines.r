# Install required packages if missing: 
# install.packages(c("ggplot2", "dplyr", "patchwork", "ggsci", "readr", "forcats"))

library(ggplot2)
library(dplyr)
library(patchwork)
library(ggsci)
library(readr)
library(forcats)

# ==========================================
# 0. GLOBAL PUBLICATION THEME
# ==========================================
theme_nature <- function() {
  theme_classic(base_size = 17, base_family = "sans") +
    theme(
      plot.title = element_text(face = "plain", size = 19, hjust = 0.5, margin = margin(b = 10)),
      axis.title.x = element_text(face = "plain", size = 17, margin = margin(t = 10)),
      axis.title.y = element_text(face = "plain", size = 17, margin = margin(r = 10)),
      axis.text = element_text(face = "plain", size = 15, color = "black"),
      legend.position = "none",
      strip.background = element_blank(),
      strip.text = element_text(face = "plain", size = 17),
      # REDUCED WHITE SPACE: Tightened margins around every individual plot
      plot.margin = margin(t = 5, r = 5, b = 5, l = 5) 
    )
}

# Paths
data_dir <- "outputs/plot_data/"
out_dir <- "outputs/figures/"

# ==========================================
# 1. BASELINE ROC CURVES (Panel A)
# ==========================================
cat("[*] Generating Baseline ROC Curves...\n")
gen_roc <- read_csv(paste0(data_dir, "genomic_roc_curve.csv"), show_col_types = FALSE)
clin_roc <- read_csv(paste0(data_dir, "clinical_roc_curve.csv"), show_col_types = FALSE)

roc_combined <- bind_rows(gen_roc, clin_roc)

plot_roc <- ggplot(roc_combined, aes(x = FPR, y = TPR, color = Modality)) +
  geom_line(size = 1.2, alpha = 0.85) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "gray50") +
  # COLOR FIX: Forcing strict mapping so Clinical is always Red and Genomic is always Blue
  scale_color_manual(values = c("Clinical" = "#ED0000", "Genomic" = "#00468B")) + 
  labs(
    title = "A. Internal Validation Baselines",
    x = "False Positive Rate",
    y = "True Positive Rate"
  ) +
  theme_nature() +
  annotate("text", x = 0.45, y = 0.25, label = "Clinical AUC: 0.884 [0.869, 0.898]", 
           color = "#ED0000", fontface = "plain", size = 5.5, hjust = 0) +
  annotate("text", x = 0.45, y = 0.15, label = "Genomic AUC: 0.764 [0.703, 0.820]", 
           color = "#00468B", fontface = "plain", size = 5.5, hjust = 0)

# ==========================================
# 2. FEATURE IMPORTANCE: LOLLIPOP PLOTS (Panel B & C)
# ==========================================
cat("[*] Generating Feature Importance Lollipops...\n")
clin_fi <- read_csv(paste0(data_dir, "clinical_feature_importance.csv"), show_col_types = FALSE) %>%
  top_n(15, Importance) %>%
  mutate(Feature = fct_reorder(Feature, Importance))

gen_fi <- read_csv(paste0(data_dir, "genomic_feature_importance.csv"), show_col_types = FALSE) %>%
  top_n(15, Importance) %>%
  mutate(Gene = fct_reorder(Gene, Importance))

plot_fi_clin <- ggplot(clin_fi, aes(x = Importance, y = Feature)) +
  geom_segment(aes(x = 0, xend = Importance, y = Feature, yend = Feature), color = "gray70") +
  geom_point(color = "#ED0000", size = 4) +
  labs(title = "B. Top Clinical Predictors", x = "Relative Importance", y = "") +
  theme_nature()

plot_fi_gen <- ggplot(gen_fi, aes(x = Importance, y = Gene)) +
  geom_segment(aes(x = 0, xend = Importance, y = Gene, yend = Gene), color = "gray70") +
  geom_point(color = "#00468B", size = 4) +
  labs(title = "C. Top Genomic Predictors", x = "Relative Importance", y = "") +
  scale_x_continuous(breaks = scales::pretty_breaks(n = 4)) +
  theme_nature()

# ==========================================
# 3. LOCO FOREST PLOT (Panel D)
# ==========================================
cat("[*] Generating LOCO Forest Plot...\n")
loco_data <- read_csv(paste0(data_dir, "loco_forest_data.csv"), show_col_types = FALSE) %>%
  mutate(Cohort = fct_reorder(Cohort, AUC))

pooled_auc <- 0.557 

plot_forest <- ggplot(loco_data, aes(x = AUC, y = Cohort)) +
  geom_vline(xintercept = 0.5, linetype = "dashed", color = "black", size = 0.8) +
  geom_vline(xintercept = pooled_auc, linetype = "dashed", color = "#00468B", size = 1) +
  geom_errorbarh(aes(xmin = AUC_Lower, xmax = AUC_Upper), height = 0.2, color = "gray30") +
  geom_point(aes(size = N_Patients), color = "#00468B") +
  scale_size_continuous(range = c(3, 8), guide = "none") + 
  labs(
    title = "D. LOCO External Validation\n(Domain Shift)",
    x = "Area Under the ROC Curve (AUC)",
    y = ""
  ) +
  theme_nature() +
  annotate("text", x = 0.70, y = 2.5, label = "I² = 85.1%\nPooled AUC = 0.557", 
           fontface = "plain", size = 5.5, hjust = 0.5, color = "gray20") +
  # NEW: Line labels (size 3.17 in ggplot mm translates exactly to 9pt font)
  annotate("text", x = 0.485, y = 8.5, label = "Random Chance", color = "black", size = 3.17, hjust = 1) +
  annotate("text", x = 0.565, y = 8.5, label = "Pooled AUC", color = "#00468B", size = 3.17, hjust = 0)

# ==========================================
# 4. COMPOSITE ASSEMBLY (Patchwork)
# ==========================================
cat("[*] Assembling Composite Figure 1...\n")

composite <- (plot_roc | plot_forest) / (plot_fi_clin | plot_fi_gen) +
  plot_layout(heights = c(1.1, 1), widths = c(1, 1.15))

# Save strictly at 300 DPI
ggsave(paste0(out_dir, "Fig1_Baselines_and_DomainShift.pdf"), plot = composite, 
       width = 15, height = 11, dpi = 300, device = cairo_pdf)
ggsave(paste0(out_dir, "Fig1_Baselines_and_DomainShift.png"), plot = composite, 
       width = 15, height = 11, dpi = 300, bg = "white")

cat("[*] DONE! Figure saved to outputs/figures/Fig1_Baselines_and_DomainShift.pdf\n")