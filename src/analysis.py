from collections import defaultdict
import numpy as np
import nltk
from nltk.corpus import wordnet as wn
import pandas as pd
from src.utils import pad_last
import re



######################################################
###############    NLP functions      ################
######################################################

# preload WordNet lemmatizer to normalize text
_wordnet_lemmatizer = nltk.WordNetLemmatizer()

"""
Normalizes a word using the WordNet lemmatizer.
    @word: the word to normalize
"""
def normalize_word(word: str, by_lbl=True) -> str:
    if by_lbl:
        word_lbl = ''
        syn = wn.synsets(word)
        if len(syn) > 0:
            word_lbl = syn[0].pos()
        if word_lbl == 'n':
           return _wordnet_lemmatizer.lemmatize( word, pos='n')
        elif word_lbl == 'v':
            _wordnet_lemmatizer.lemmatize( word, pos='v')
        else:
            return word
    return _wordnet_lemmatizer.lemmatize(
                _wordnet_lemmatizer.lemmatize( word, pos='n')
                , pos='v')

"""
Extracts words from a prompt.
    @text: the text to exctract an ordered set of words from
Returns
    @all_words: normalized set of words, including participles etc.
    @key_words: nouns, adjectives, verbs
"""
def filter_text(text: str) -> list[str]:
    all_words = []
    key_words = []
    key_word_phrases = [[]]

    for word in nltk.word_tokenize(text):
        if word.isalpha() | word.isnumeric():
            word = word.lower()
            word = re.sub(r'[^\w\s]+$', '', word)
            # word_lbl = nltk.pos_tag([word])[0][1]
            key_words.append(False)
            if True or ((word not in nltk.corpus.stopwords.words("english")) and 
               (word_lbl not in ['DT', 'IN', 'CC'])):
                word = normalize_word(word)
                key_word_phrases[-1].append(word)
                key_words[-1] = True
            all_words.append(word)
        else:
            key_word_phrases.append([])
    return all_words, key_words, [kwp for kwp in key_word_phrases if len(kwp) > 0]

"""
Compute Intersection over Union (IoU) of two passages, after normalizing the texts.
    @p1, @p2: the 2 passages to compute IoU over
    @key_words:
        == True  ---> use only nouns, adjectives, verbs for IoU
        == False ---> use all normalized text
"""
def IoU(p1: str, p2: str, key_words=True) -> float:
    word_list_1 = set(filter_text(p1)[int(key_words)])
    word_list_2 = set(filter_text(p2)[int(key_words)])
    intersection = word_list_1 & word_list_2
    union = word_list_1 | word_list_2
    return len(intersection) / len(union)


######################################################
#############    Data preprocessing      #############
######################################################

"""
Computes data about word usage from a pandas.DataFrame containing prompts.
    @user_summary_df: output from `src.load_data.load_saved_data()`
    @prompt_df: output from `src.load_data.extract_prompt_summaries()`
"""
def word_stats(user_summary_df: pd.DataFrame, prompt_df: pd.DataFrame) -> pd.DataFrame:
    ret_word_stats = []
    for wi, dfi in prompt_df.groupby('word'):
        ret_word_stats.append({
            'word': wi,
            'n_prompts': len(dfi.prompt_id.unique()),
            'n_users': len(dfi.user.unique()),
            'frac_in_pos_prompt': (dfi.prompt=='+').mean(),
            'frac_in_neg_prompt': (dfi.prompt=='-').mean(),
            'in_frac_pos_prompt': len(dfi[dfi.prompt=='+'].prompt_id.unique()) / len(user_summary_df),
            'in_frac_neg_prompt': len(dfi[dfi.prompt=='-'].prompt_id.unique()) / len(user_summary_df),
            'score': dfi.score.mean(),
            'score_min': dfi.score.min(),
            'score_max': dfi.score.max(),
            'prompt_len': dfi.prompt_len.mean()
        })
    return pd.DataFrame(ret_word_stats)


"""
Analyzes a single user on a target image.
    @user_data, @metadata: outputs from `src.load_data.load_saved_data()`
    @user_id: hash value used for user IDs
    @target: targetID [integer >= 1]
"""
def analyze_user_on_target(user_data: dict, metadata: dict, user_id: str, target=1) -> dict[str]:
    df = user_data[user_id][
        (~pd.isna(user_data[user_id].prompt_hash)) &
        (user_data[user_id].target == target)]

    scores = np.array(list(df.score))
    p_prompts = []
    n_prompts = []
    for pi in df.prompt_hash:
        p_prompts.append(metadata[pi]['prompt+'])
        n_prompts.append(metadata[pi]['prompt-'])
        
    p_prompt_len = np.array([len(pi) for pi in p_prompts])
    n_prompt_len = np.array([len(pi) for pi in n_prompts])
    
    prompts = [f"{p_prompt} ; {n_prompt}" for p_prompt,n_prompt in zip(p_prompts, n_prompts)]
    prompt_word_list = [set(filter_text(pi)[1]) for pi in prompts]
    Iow1_mat = np.array(
            [[len(word_list_1 & word_list_2) / len(word_list_1)
                   for word_list_1 in prompt_word_list]
                       for word_list_2 in prompt_word_list
    ])
    Iow2_mat = np.array(
            [[len(word_list_1 & word_list_2) / len(word_list_2)
                   for word_list_1 in prompt_word_list]
                       for word_list_2 in prompt_word_list
    ])
    IoU_mat = np.array(
            [[len(word_list_1 & word_list_2) / len(word_list_1 | word_list_2)
                   for word_list_1 in prompt_word_list]
                       for word_list_2 in prompt_word_list
    ])

    difference_from_prev_prompt = 1 - np.array([0] + [IoU_mat[mi][mi+1] for mi in range(len(IoU_mat)-1)])

    # Defining new prompt as one that differs from the immediately previous prompt by >= 0.5 IoU
    n_new_prompts = np.cumsum(difference_from_prev_prompt>=0.5)
        
    return {'p_prompt_len': p_prompt_len,
            'n_prompt_len': n_prompt_len,
            'scores': scores,
            'Iow1_mat': Iow1_mat,
            'Iow2_mat': Iow2_mat,
            'IoU_mat': IoU_mat,
            'difference_from_prev_prompt': difference_from_prev_prompt,
            'n_new_prompts': n_new_prompts,
            }


"""
Analyzes a list of users across target images. Returns a dictionary where the keys are user IDs, and
the values are a tuple of the outputs from `analyze_user_on_target()`.
    @user_data, @metadata: outputs from `src.load_data.load_saved_data()`
    @user_id: hash value used for user IDs
    @targets: list of targets
"""
def analyze_users(user_data: dict, metadata: dict, user_ids: list[str], targets: list) -> dict[str, dict]:
    users_analysis = {}
    for user_id in user_ids:
        users_analysis[user_id] = dict()
        for target in targets:
            users_analysis[user_id][target] = analyze_user_on_target(user_data, metadata, user_id, target)

    return users_analysis
    
"""
Given a dictionary of users, iterate over the values and compute average and standard deviation where possible.

Assumes each user data is stored as a dictionary mapping target_ids to a dictionary of summary data. Averages 
will be computed against this summary data.
    --> [{user_1: 
            {target_1: 
                {data_1: value_1, data_2, value_2, ...}, 
            target_2: ...}, 
          user_2: ...},
          ...
        ]

Returns dictionary of summarizing users:
if aggregate_across_targets:
    --> pd.DataFrame([{summary_value: data_1, mean: mean(value_1), std: std(value_1), ...},
        ...]
else:
    --> pd.DataFrame([{summary_value: data_1, target: target_1, mean: mean(value_1), std: std(value_1), ...},
        ...])
"""
def summarize_users(users: dict[str, dict], padding=True, truncate=False, aggregate_across_targets=False) -> pd.DataFrame:
    assert not (padding and truncate), "At most 1 of `padding` and `trucnate` option can be true."
    if aggregate_across_targets: 
        value_lists = defaultdict(list)
        max_len = defaultdict(lambda: 0)
        min_len = defaultdict(lambda: 1e6)
    else: 
        value_lists = defaultdict(lambda: defaultdict(list))
        max_len = defaultdict(lambda: defaultdict(lambda: 0))
        min_len = defaultdict(lambda: defaultdict(lambda: 1e6))


    # Iterate across users -> targets -> user data summaries
    for _,user_data in users.items():
        for target_id, target_data in user_data.items():
            for data_k, data_v in target_data.items():
                # Pre-process data to normalize it
                if not isinstance(data_v, np.ndarray):
                    continue
                if len(data_v.shape) > 1:
                    continue

                if aggregate_across_targets: 
                    value_lists[data_k].append(data_v)
                    max_len[data_k] = max(max_len[data_k], len(data_v))
                    min_len[data_k] = min(min_len[data_k], len(data_v))
                else: 
                    value_lists[target_id][data_k].append(data_v)
                    max_len[target_id][data_k] = max(max_len[target_id][data_k], len(data_v))
                    min_len[target_id][data_k] = min(min_len[target_id][data_k], len(data_v))
    # Apply padding or truncate the values depending on the options
    if padding or truncate:
        if aggregate_across_targets:
            for _k,_v in value_lists.items():
                if padding: value_lists[_k] = [pad_last(_vi, max_len[_k] - len(_vi)) for _vi in _v]
                if truncate: value_lists[_k] = [_vi[:min_len[_k]] for _vi in _v]
        else:
            for _k,_v in value_lists.items():
                for _k2,_v2 in _v.items():
                    if padding: value_lists[_k][_k2] = [pad_last(_v2i, max_len[_k][_k2] - len(_v2i)) for _v2i in _v2]
                    if truncate: value_lists[_k][_k2] = [_v2i[:min_len[_k][_k2]] for _v2i in _v2]

    # Compute mean and std. dev.
    if aggregate_across_targets: 
        summarized_users = [{
                'summary_value': _k, 
                'mean': np.nanmean(_v, axis=0), 
                'median': np.nanmedian(_v, axis=0),
                'std': np.nanstd(_v, axis=0), 
                '10th': np.nanquantile(_v, q=0.1, axis=0), 
                '90th': np.nanquantile(_v, q=0.9, axis=0)
            } for _k,_v in value_lists.items()
        ]
    else: 
        summarized_users = [{
                'summary_value': _k, 
                'target': _k2, 
                'mean': np.nanmean(_v2, axis=0), 
                'median': np.nanmedian(_v2, axis=0),
                'std': np.nanstd(_v2, axis=0), 
                '10th': np.nanquantile(_v2, q=0.1, axis=0), 
                '90th': np.nanquantile(_v2, q=0.9, axis=0)
            } for _k,_v in value_lists.items()
                for _k2,_v2 in _v.items()
        ]
    return pd.DataFrame(summarized_users)