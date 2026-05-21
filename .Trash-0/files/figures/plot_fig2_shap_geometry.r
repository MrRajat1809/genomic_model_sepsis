# Install required packages if missing
# install.packages(c("ggplot2", "dplyr", "patchwork", "readr", "forcats", "ggbeeswarm"))

library(ggplot2)
library(dplyr)
library(patchwork)
library(readr)
library(forcats)
library(ggbeeswarm)

# ==========================================
# 0. GLOBAL PUBLICATION THEME
# ==========================================
theme_nature_shap <- function() {
  theme_classic(base_size = 17, base_family = "sans") +
    theme(
      plot.title = element_text(face = "plain", size = 18, hjust = 0.5, margin = margin(b = 15)),
      axis.title.x = element_text(face = "plain", size = 17, margin = margin(t = 12)),
      axis.title.y = element_text(face = "plain", size = 17, margin = margin(r = 12)),
      axis.text = element_text(face = "plain", size = 15, color = "black"),
      legend.position = "right",
      # Margins slightly expanded on the right to prevent legend truncation
      plot.margin = margin(t = 10, r = 20, b = 10, l = 10),
      panel.grid.major.y = element_line(color = "gray90", linetype = "dotted")
    )
}

# Authentic SHAP Colors
shap_blue <- "#1E88E5"
shap_red <- "#ff0052"

# Paths
data_dir <- "outputs/plot_data/"
out_dir <- "outputs/figures/"

# ==========================================
# 1. LOAD AND PREP DATA
# ==========================================
cat("[*] Loading SHAP Data for R...\n")

clin_shap <- read_csv(paste0(data_dir, "shap_clinical_export.csv"), show_col_types = FALSE)
gen_shap <- read_csv(paste0(data_dir, "shap_genomic_export.csv"), show_col_types = FALSE)

clin_shap <- clin_shap %>%
  group_by(Feature) %>%
  mutate(MeanAbsShap = mean(abs(SHAP_Value))) %>%
  ungroup() %>%
  mutate(Feature = fct_reorder(Feature, MeanAbsShap))

gen_shap <- gen_shap %>%
  group_by(Feature) %>%
  mutate(MeanAbsShap = mean(abs(SHAP_Value))) %>%
  ungroup() %>%
  mutate(Feature = fct_reorder(Feature, MeanAbsShap))

# ==========================================
# 2. LEGEND FORMATTING (The Fix)
# ==========================================
# We build a custom colorbar guide so "High" and "Low" aren't cut off
# and "Feature value" is rotated perfectly.
shap_colorbar <- guide_colorbar(
  title = "Feature value",
  title.position = "right",
  title.theme = element_text(angle = 270, hjust = 0.5, size = 15, margin = margin(l = 10)),
  label.theme = element_text(size = 13),
  barwidth = unit(0.5, "cm"),
  barheight = unit(8, "cm"),
  ticks = FALSE
)

# ==========================================
# 3. BUILD PANEL A: CLINICAL
# ==========================================
cat("[*] Generating Clinical SHAP Distributions...\n")
plot_clin <- ggplot(clin_shap, aes(x = SHAP_Value, y = Feature, color = Feature_Value_Scaled)) +
  geom_vline(xintercept = 0, color = "gray50", size = 0.8) +
  geom_quasirandom(groupOnX = FALSE, varwidth = TRUE, size = 1.5, alpha = 0.8, stroke = 0) +
  scale_color_gradientn(
    colors = c(shap_blue, "#a05195", shap_red),
    breaks = c(0, 1),
    labels = c("Low", "High"),
    na.value = "gray50",
    guide = shap_colorbar
  ) +
  labs(
    title = "SHAP Summary: Clinical Predictors of Sepsis Onset",
    x = "SHAP Value (Impact on model output)",
    y = ""
  ) +
  theme_nature_shap()

# ==========================================
# 4. BUILD PANEL B: GENOMIC
# ==========================================
cat("[*] Generating Genomic SHAP Distributions...\n")
plot_gen <- ggplot(gen_shap, aes(x = SHAP_Value, y = Feature, color = Feature_Value_Scaled)) +
  geom_vline(xintercept = 0, color = "gray50", size = 0.8) +
  geom_quasirandom(groupOnX = FALSE, varwidth = TRUE, size = 1.5, alpha = 0.8, stroke = 0) +
  scale_color_gradientn(
    colors = c(shap_blue, "#a05195", shap_red),
    breaks = c(0, 1),
    labels = c("Low", "High"),
    na.value = "gray50",
    guide = shap_colorbar
  ) +
  labs(
    title = "SHAP Summary: Genetic Predictors of Sepsis Mortality",
    x = "SHAP Value (Impact on model output)",
    y = ""
  ) +
  theme_nature_shap()

# ==========================================
# 5. COMPOSITE ASSEMBLY (Patchwork)
# ==========================================
cat("[*] Assembling Composite Figure 2...\n")

composite <- plot_clin + plot_gen +
  plot_layout(ncol = 2, guides = "collect") & theme(legend.position = "right")

# Save strictly at 300 DPI
ggsave(paste0(out_dir, "Fig2_Interpretability_Geometry.pdf"), plot = composite, 
       width = 16, height = 8, dpi = 300, device = cairo_pdf)
ggsave(paste0(out_dir, "Fig2_Interpretability_Geometry.png"), plot = composite, 
       width = 16, height = 8, dpi = 300, bg = "white")

cat("[*] DONE! Figure saved to outputs/figures/Fig2_Interpretability_Geometry.pdf\n")