# -- High-level statistics etc. --
# Imports
from src.load_data import load_all_data
from src.constants import PLOT_DIR, PRELOADED_USER_SUMMARIES, PRELOADED_TEXT_EMBED, PRELOADED_IMAGE_EMBED
import numpy as np
# Constants
plot_dir = f"{PLOT_DIR}/{__file__.split('/')[-1].split('.')[0]}"
# Load data
dataset, (user_summaries, text_embeddings, image_embeddings) = load_all_data(force=True)
# Save data
user_summaries.to_csv(PRELOADED_USER_SUMMARIES, index=False)
np.save(PRELOADED_TEXT_EMBED, text_embeddings)
np.save(PRELOADED_IMAGE_EMBED, image_embeddings)