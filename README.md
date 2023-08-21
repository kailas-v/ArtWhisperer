# ArtWhisperer Dataset

## Overview
The ArtWhisperer Dataset captures human interactions with an image-generation AI model (variants of the Stable Diffusion model) as they attempt to generate a specified target image. This dataset includes the trajectory of prompts and images generated, the target image, and a score trajectory for each user-AI interaction. 


***More details on the dataset can be found in our paper:*** [ArtWhisperer: A Dataset for Characterizing Human-AI Interactions in Artistic Creations
](https://arxiv.org/abs/2306.08141)


### Example of code use
Our dataset is available on [Hugging Face](https://huggingface.co/datasets/kailasv/ArtWhisperer). Loading from Hugging Face requires installation of the datasets library (`pip install datasets`). Example code is below:
```python
from datasets import load_dataset

dataset = load_dataset("kailasv/ArtWhisperer")
```

### Description of dataset

The dataset contains two splits `train` and `validation`. Details on how these are defined are described in our paper. The `validation` dataset contains additional information where the humans interacting with the model also gave their own ratings for how close their generated images are to the target image.

Each data instance contains several entries:

#### Interaction ID
- `user_id`: string identifying the user
- `target_id`: string identifying the target image
#### Target image info
- `target_image`: a PIL Image of the target image the user was tasked with generating
- `target_positive_prompt`: Description of the target image
- `target_negative_prompt`: Negative description of the target image (for all target images we used, this is an empty string)
- `target_image_embedding`: CLIP image embedding of `target_image`
- `target_positive_text_embedding`: CLIP text embedding of `target_positive_prompt` 
- `target_negative_text_embedding`: CLIP text embedding of `target_positive_prompt` 
#### Generated image info
- `generated_image`: a PIL Image of the user-generated image
- `generated_positive_prompt`: user-submitted prompt for generating `generated_image`
- `generated_negative_prompt`: user-submitted negative prompt for generating `generated_image`
- `generated_image_embedding`: CLIP image embedding of `generated_image`
- `generated_positive_text_embedding`: CLIP text embedding of `generated_positive_prompt` 
- `generated_negative_text_embedding`: CLIP text embedding of `generated_negative_prompt` 
#### Additional information about the interaction
- `ai_model_name`: name of the AI model used for this interaction (either 'SDv2.1' or 'SDv1.5')
- `trajectory_index`: ordering for the given interaction (indexing starts from 1 and restarts for each `user_id`, `target_id` pair)
- `score`: automated scoring to assess how similar `target_image` and `generated_image` are (bewteen 0 and 100)
- `human_rating`: user's rating for similarity bewteen `target_image` and `generated_image` (bewteen 0 and 100)
- `time_taken`: duration in seconds the user took to write/update their prompts
- `filtered_image`: whether the user-generated image triggered an NSFW-filter (if it did, `generated_image` will be a black image)

## Citation
If you find this work useful or use this dataset in your research, please cite:
```
@article{vodrahalli2023artwhisperer,
  title={ArtWhisperer: A Dataset for Characterizing Human-AI Interactions in Artistic Creations},
  author={Vodrahalli, Kailas and Zou, James},
  journal={arXiv preprint arXiv:2306.08141},
  year={2023}
}
```

## Contact
If you have any questions, please feel free to email the authors.
