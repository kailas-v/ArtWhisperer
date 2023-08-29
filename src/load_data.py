import numpy as np
import os
import pandas as pd
import src.utils as utils
import src.analysis as analysis
from datasets import load_dataset, concatenate_datasets
from src.constants import (
    # General constants
    PRELOADED_USER_SUMMARIES, PRELOADED_TEXT_EMBED, PRELOADED_IMAGE_EMBED,
    # Analysis constants
    MIN_SCORE_CUTOFF, MIN_TARGET_SCORE_CUTOFF, MIN_NUM_USERS
)
from collections import defaultdict
from tqdm import tqdm

def prefilter_data(user_summaries, min_score_cutoff=MIN_SCORE_CUTOFF):
    filter_user_id = []
    # Filter-out "test" users (me) and users with very low score
    for user_id,df in user_summaries.groupby('user'):
        df = df[~(df.pos_prompt.isna())]
        n_empty = sum([len(pi.strip())==0 for pi in df.pos_prompt])
        n_total = len(df)
        median_prompt_len = df.pos_prompt_len.median()
        if (n_total == 0) or ((n_empty/n_total > 0.4) and median_prompt_len < 20) or \
            df.score.max() < min_score_cutoff:
            filter_user_id.append(user_id)
    user_summaries = user_summaries[~user_summaries.user.isin(filter_user_id)]
    # Filter-out target images that had too few users or too low score
    filter_target_id = []
    for target,df in user_summaries.groupby('target'):
        n_user = len(df.user.unique())
        if (n_user < MIN_NUM_USERS) or (df.score.max() < MIN_TARGET_SCORE_CUTOFF):
            filter_target_id.append(target)
    user_summaries = user_summaries[~user_summaries.target.isin(filter_target_id)]
    return user_summaries

def prefilter_data_count(user_summaries, min_score_cutoff=MIN_SCORE_CUTOFF):
    base_count = len(user_summaries)
    original_count = base_count
    filter_user_id = []
    # Filter-out "test" users (me) and users with very low score
    for user_id,df in user_summaries.groupby('user'):
        df = df[~df.pos_prompt.isna()]
        n_test = sum(['test' in pi.lower() for pi in df.pos_prompt])
        n_empty = sum([len(pi.strip())==0 for pi in df.pos_prompt])
        n_total = len(df)
        median_prompt_len = df.pos_prompt_len.median()
        if (n_total == 0) or (((n_test/n_total > 0.4) or (n_empty/n_total > 0.4)) and median_prompt_len < 20):
            filter_user_id.append(user_id)
    user_summaries = user_summaries[~user_summaries.user.isin(filter_user_id)]
    new_count = len(user_summaries)
    print(f"Filtered for being test intercations: {base_count - new_count} / {original_count}")
    base_count = new_count
    for user_id,df in user_summaries.groupby('user'):
        df = df[~df.pos_prompt.isna()]
        if df.score.max() < min_score_cutoff:
            filter_user_id.append(user_id)
    user_summaries = user_summaries[~user_summaries.user.isin(filter_user_id)]
    new_count = len(user_summaries)
    print(f"Filtered for being low max scores: {base_count - new_count} / {original_count}")
    base_count = new_count
    # Filter-out target images that had too few users or too low score
    filter_target_id = []
    for target,df in user_summaries.groupby('target'):
        n_user = len(df.user.unique())
        if (n_user < MIN_NUM_USERS) or (df.score.max() < MIN_TARGET_SCORE_CUTOFF):
            filter_target_id.append(target)
    user_summaries = user_summaries[~user_summaries.target.isin(filter_target_id)]
    new_count = len(user_summaries)
    print(f"Filtered target images: {base_count - new_count} / {original_count}")
    base_count = new_count
    return user_summaries

def load_text_trajectories(dataset, user_summaries, text_embeddings, only_pos=True):
    DEFAULT_EMBEDDING = {'CLIP': np.zeros(1024)}
    def load_text_embedding(x, embedding_type):
        if x not in text_embeddings:
            return DEFAULT_EMBEDDING[embedding_type]
        x = text_embeddings[x]
        if 'p_text_features' in x:
            x = x['p_text_features'][0]
        return x.get(embedding_type, DEFAULT_EMBEDDING[embedding_type])
    all_trajectories = {}
    embedding_types = list(list(text_embeddings.values())[0].keys())
    for embedding_type in embedding_types:
        trajectories = defaultdict(list)
        for (user, target), df in user_summaries.groupby(['user', 'target']):
            data_split = list(df['data_split'])[0]
            p_embeddings = df['pos_prompt'].apply(lambda x: load_text_embedding(x, embedding_type))
            n_embeddings = df['neg_prompt'].apply(lambda x: load_text_embedding(x, embedding_type))
            model_used = list(df['model_used'])[0]
            idxs = [i for i,(pi,ni) in enumerate(zip(p_embeddings, n_embeddings)) if ((pi is not None) or (ni is not None))]
            p_embeddings = np.array(p_embeddings)[idxs]
            n_embeddings = np.array(n_embeddings)[idxs]
            if only_pos:
                trajectory = [pi[None,:] for pi,ni in zip(p_embeddings, n_embeddings)]
            else:
                trajectory = [np.concatenate((pi,ni))[None,:] for pi,ni in zip(p_embeddings, n_embeddings)]
            score_trajectory = np.array(df['score'])[idxs]
            rating_trajectory = np.array(df['rating'])[idxs]
            p_prompts = np.array(df['pos_prompt'])[idxs]
            n_prompts = np.array(df['neg_prompt'])[idxs]
            map_to_datasets = np.array(df['map_to_dataset'])[idxs]

            if len(trajectory) == 1:
                trajectory = trajectory[0]
            elif len(trajectory) > 1:
                trajectory = np.concatenate(trajectory)
            else:
                continue
            trajectories[target].append(dict(
                user=user,
                model_used=model_used, 
                data_split=data_split,
                trajectory=trajectory, 
                score_trajectory=score_trajectory, 
                p_prompts=p_prompts, 
                n_prompts=n_prompts, 
                map_to_datasets=map_to_datasets,
                rating_trajectory=rating_trajectory,
                target_prompt=dataset[int(map_to_datasets[0])]['target_positive_prompt'],
            ))
        all_trajectories[embedding_type] = trajectories
    return all_trajectories

def load_image_trajectories(dataset, user_summaries, image_embeddings, only_pos=True):
    check_is_str = lambda x: isinstance(x, str) and x
    all_trajectories = {}
    embedding_types = ['CLIP']
    for embedding_type in embedding_types:
        trajectories = defaultdict(list)
        for (user, target), df in user_summaries.groupby(['user', 'target']):
            i_embeddings = df['prompt_hash'].apply(lambda x: image_embeddings.get(x, None))
            p_prompts = df['pos_prompt']
            n_prompts = df['neg_prompt']
            idxs = [i for i,(i_embedding,pi,ni) in enumerate(zip(i_embeddings, p_prompts, n_prompts)) if ((i_embedding is not None) and (check_is_str(pi)) and ((only_pos or check_is_str(ni))))]
            trajectory = np.array([i_embedding[None,:] for i_embedding in i_embeddings])[idxs]
            score_trajectory = np.array(df['score'])[idxs]
            p_prompts = np.array(df['pos_prompt'])[idxs]
            n_prompts = np.array(df['neg_prompt'])[idxs]
            map_to_datasets = np.array(df['map_to_dataset'])[idxs]
            if len(trajectory) == 1:
                trajectory = trajectory[0]
            elif len(trajectory) > 1:
                trajectory = np.concatenate(trajectory)
            else:
                continue
            trajectories[target].append(dict(user=user, trajectory=trajectory, score_trajectory=score_trajectory, p_prompts=p_prompts, n_prompts=n_prompts, map_to_datasets=map_to_datasets))
        all_trajectories[embedding_type] = trajectories
    return all_trajectories

def get_dist(x1,x2):
    return np.sqrt(np.mean(np.linalg.norm(x1-x2,axis=1)**2))

def get_trajectory_distances(dataset, all_trajectories, embeddings, min_n_prompts=None, max_n_prompts=None, is_image=False):
    all_trajectory_analysis = {}
    min_n_prompts = (min_n_prompts if min_n_prompts is not None else 2)
    targets = list(all_trajectories['CLIP'].keys())
    for embedding_type,trajectories_ in all_trajectories.items():
        all_trajectory_analysis[embedding_type] = {}
        for target in targets:
            if is_image:
                target_embedding = embeddings[f'target-{target}']
            else:
                actual_prompt = all_trajectories['CLIP'][target][0]['target_prompt']
                target_embedding = embeddings[actual_prompt][embedding_type]
            trajectories_data = trajectories_[target]

            # Trajectories and scores
            trajectories = [ti['trajectory'][:(max_n_prompts if max_n_prompts is not None else len(ti['trajectory']))] 
                                for ti in trajectories_data if len(ti['trajectory']) >= min_n_prompts]
            if len(trajectories) == 0:
                continue
            scores = [ti['score_trajectory'][:(max_n_prompts if max_n_prompts is not None else len(ti['score_trajectory']))] 
                        for ti in trajectories_data if len(ti['trajectory']) >= min_n_prompts]
            # First, last, and best prompts in trajectory
            trajectories_f = [ti[0] for ti in trajectories]
            trajectories_l = [ti[-1] for ti in trajectories]
            trajectories_b = [ti[np.argmax(scores[i])] for i,ti in enumerate(trajectories)]
            scores_f = [ti[0] for ti in scores]
            scores_l = [ti[-1] for ti in scores]
            scores_b = [ti[np.argmax(scores[i])] for i,ti in enumerate(scores)]

            # Fake users
            fake_users_n = [len(ti) for ti in trajectories]
            all_prompt_embeddings = np.concatenate(trajectories, axis=0)

            # Calculate distances
            avg_trajectory = [np.mean(ti, axis=0) for ti in trajectories]
            dist_from_avg = [get_dist(ti,ati) for ti,ati in zip(trajectories, avg_trajectory)]
            dist_from_gt = [get_dist(ti,target_embedding) for ti in trajectories]

            rng = utils.rng_from_obj(target)
            fake_trajectories = [all_prompt_embeddings[rng.choice(len(all_prompt_embeddings), size=ni, replace=True)] for ni in fake_users_n]
            fake_avg_trajectory = [np.mean(ti, axis=0) for ti in fake_trajectories]
            fake_dist_from_avg = [get_dist(ti,ati) for ti,ati in zip(fake_trajectories, fake_avg_trajectory)]
            fake_dist_from_gt = [get_dist(ti,target_embedding) for ti in fake_trajectories]


            avg_trajectory_f = np.mean(trajectories_f, axis=0)
            avg_trajectory_l = np.mean(trajectories_f, axis=0)
            avg_trajectory_b = np.mean(trajectories_b, axis=0)
            dist_from_avg_f = [np.linalg.norm(ti-avg_trajectory_f) for ti in trajectories_f]
            dist_from_avg_l = [np.linalg.norm(ti-avg_trajectory_l) for ti in trajectories_l]
            dist_from_avg_b = [np.linalg.norm(ti-avg_trajectory_b) for ti in trajectories_b]
            dist_from_gt_f = [np.linalg.norm(ti-target_embedding) for ti in trajectories_f]
            dist_from_gt_l = [np.linalg.norm(ti-target_embedding) for ti in trajectories_l]
            dist_from_gt_b = [np.linalg.norm(ti-target_embedding) for ti in trajectories_b]
            

            all_trajectory_analysis[embedding_type][target] = dict(
                # Distance from average prompt
                dist_from_avg=dist_from_avg,
                dist_from_gt=dist_from_gt,
                # Distance from fake users prompts
                fake_dist_from_avg=fake_dist_from_avg,
                fake_dist_from_gt=fake_dist_from_gt,
                # Distance from individual prompts
                dist_from_avg_f=dist_from_avg_f,
                dist_from_gt_f=dist_from_gt_f,
                dist_from_avg_l=dist_from_avg_l,
                dist_from_gt_l=dist_from_gt_l,
                dist_from_avg_b=dist_from_avg_b,
                dist_from_gt_b=dist_from_gt_b,
                # Scores
                scores=scores,
                scores_f=scores_f,
                scores_l=scores_l,
                scores_b=scores_b,
            )
    return all_trajectory_analysis


"""
Loads saved data from game and preprocess to more efficient format for analysis.
"""
def load_np_from_dataset(v):
    return np.array(v['value'])
def load_saved_data(all_data):
    # Load ArtWhisperer Dataset
    n_train = len(all_data['train'])
    dataset = concatenate_datasets([all_data['train'], all_data['validation']])

    # Iterate across dataset and reformat files
    #   (but not the actual images, just the image paths)
    user_summaries = []
    text_embeddings = {}
    image_embeddings = {}

    best_score = defaultdict(lambda: 0)
    best_rating = defaultdict(lambda: 0)

    for i, d in tqdm(enumerate(dataset), total=len(dataset)):
        data_split = "ArtWhisperer" if i < n_train else "ArtWhisperer-Validation"
        # Text Embedding
        text_embeddings[d['target_positive_prompt']] = {'CLIP': load_np_from_dataset(d['target_positive_text_embedding'])}
        text_embeddings[d['target_negative_prompt']] = {'CLIP': load_np_from_dataset(d['target_negative_text_embedding'])}
        text_embeddings[d['generated_positive_prompt']] = {'CLIP': load_np_from_dataset(d['generated_positive_text_embedding'])}
        text_embeddings[d['generated_negative_prompt']] = {'CLIP': load_np_from_dataset(d['generated_negative_text_embedding'])}

        # Image Embedding
        image_embeddings[d['target_id']] = load_np_from_dataset(d['target_image_embedding'])
        image_embeddings[i] = load_np_from_dataset(d['generated_image_embedding'])

        # User Summaries
        user_key = (d['user_id'], d['target_id'])
        user_summaries.append({
            'user': d['user_id'],
            'target': d['target_id'],
            'map_to_dataset': i,
            'trajectory_index': d['trajectory_index'],
            'data_split': data_split,
            'model_used': d['ai_model_name'],
            'time_taken': d['time_taken'],
            'score': d['score'],
            'improved_score': d['score'] > best_score[user_key],
            'rating': -1 if np.isnan(d['human_rating']) else d['human_rating'],
            'improved_rating': False if np.isnan(d['human_rating']) else d['human_rating'] > best_rating[user_key],
            'pos_prompt_len': len(d['generated_positive_prompt']),
            'neg_prompt_len': len(d['generated_negative_prompt']),
            'pos_prompt': d['generated_positive_prompt'],
            'neg_prompt': d['generated_negative_prompt'],
            "target_labels": {
                'Famous person?': d['Famous person?'],
                'Famous landmark?': d['Famous landmark?'],
                'Manmade?': d['Manmade?'],
                'People?': d['People?'],
                'Real image?': d['Real image?'],
                'AI image?': d['AI image?'],
                'Art?': d['Art?'],
                'Nature?': d['Nature?'],
                'City?': d['City?'],
                'Fantasy?': d['Fantasy?'],
                'Sci-fi or space?': d['Sci-fi or space?'],
            },
        })

        # Update best score and rating
        best_score[user_key] = max(best_score[user_key], user_summaries[-1]['score'])
        best_rating[user_key] = max(best_rating[user_key], user_summaries[-1]['rating'])

    # - convert to df
    user_summaries = pd.DataFrame(user_summaries)

    # Return the two loaded data dicts and the summary data df
    return user_summaries, text_embeddings, image_embeddings

def load_all_data(force=False):
    dataset = load_dataset(path="kailasv/ArtWhisperer")
    if (not force) and ( os.path.exists(PRELOADED_TEXT_EMBED) and 
                         os.path.exists(PRELOADED_IMAGE_EMBED) and 
                         os.path.exists(PRELOADED_USER_SUMMARIES)):
        text_embeddings = np.load(PRELOADED_TEXT_EMBED, allow_pickle=True).item()
        image_embeddings = np.load(PRELOADED_IMAGE_EMBED, allow_pickle=True).item()
        user_summaries = pd.read_csv(PRELOADED_USER_SUMMARIES)
    else:
        (user_summaries, text_embeddings, image_embeddings) = load_saved_data(dataset)

    dataset = concatenate_datasets([dataset['train'], dataset['validation']])
    return dataset, (user_summaries, text_embeddings, image_embeddings)

######################################################
#############    Data preprocessing      #############
######################################################

"""
Extracts summary data on prompt usage, normalizing prompt text.
    @user_summary_df: output from `load_saved_data()`
"""
def extract_prompt_summaries(user_summary_df: pd.DataFrame, n_gram=1, kwp=False) -> tuple[dict, dict]:
    prompts_p = []
    prompts_n = []
    for i,r in user_summary_df.iterrows():
        aw_pos, kw_pos, kwp_pos = analysis.filter_text(r.pos_prompt)
        aw_neg, kw_neg, kwp_neg = analysis.filter_text(r.neg_prompt)
        if kwp:
            for kwp_i in kwp_pos:
                prompts_p.extend([{'word': ' '.join(kwp_i[i:i+n_gram]), 'prompt': '+', 'score': r.score, 
                                'prompt_len': r.pos_prompt_len, 'prompt_id':i, 'user':r.user, 'key':1} 
                                    for i in range(0,max(1,len(kwp_i)-n_gram+1))])
            for kwp_i in kwp_neg:
                prompts_n.extend([{'word': ' '.join(kwp_i[i:i+n_gram]), 'prompt': '-', 'score': r.score, 
                            'prompt_len': r.neg_prompt_len, 'prompt_id':i, 'user':r.user, 'key':1} 
                                for i in range(0,max(1,len(kwp_i)-n_gram+1))])
        else:
            prompts_p.extend([{'word': ' '.join(aw_pos[i:i+n_gram]), 'prompt': '+', 'score': r.score, 
                            'prompt_len': r.pos_prompt_len, 'prompt_id':i, 'user':r.user, 'key':np.mean(kw_pos[i:i+n_gram])} 
                                for i in range(0,max(1,len(aw_pos)-n_gram+1))])
            prompts_n.extend([{'word': ' '.join(aw_neg[i:i+n_gram]), 'prompt': '-', 'score': r.score, 
                            'prompt_len': r.neg_prompt_len, 'prompt_id':i, 'user':r.user, 'key':np.mean(kw_neg[i:i+n_gram])} 
                                for i in range(0,max(1,len(aw_neg)-n_gram+1))])
    prompts_p = pd.DataFrame(prompts_p)
    prompts_n = pd.DataFrame(prompts_n)
    return prompts_p, prompts_n
