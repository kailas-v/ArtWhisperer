import os

# Analysis-related constants
MIN_SCORE_CUTOFF = 20 # for user
MIN_TARGET_SCORE_CUTOFF = 50 # for image overall
MIN_NUM_USERS = 1#5
MAX_N_CLUSTERS = 10
PCA_DIM = 20

# Plot and Analysis saving paths
ROOT_DIR = "."
INTERMEDIATE_FILE_DIR = "preprocess_dir"
PRELOADED_USER_SUMMARIES = f"{INTERMEDIATE_FILE_DIR}/0_preloaded_user_summaries.csv"
PRELOADED_TEXT_EMBED = f"{INTERMEDIATE_FILE_DIR}/0_preloaded_text_embeddings.npy"
PRELOADED_IMAGE_EMBED = f"{INTERMEDIATE_FILE_DIR}/0_preloaded_image_embeddings.npy"
PLOT_DIR = "Figures"

os.makedirs(INTERMEDIATE_FILE_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)