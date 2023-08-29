# -- Cluster analysis and diversity of clusters --
# Imports
from src.load_data import load_all_data, prefilter_data, load_text_trajectories, get_trajectory_distances
from src.constants import PLOT_DIR, MAX_N_CLUSTERS, PCA_DIM
from src.plot_utils import save_fig, new_fig, setup_axes, PLOT_COLORS, save_with_cropped_whitespace
from src.stats_utils import bootstrap_mean_interval
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mtick
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import os

BINS = [(0,21), (21,41), (41,61), (61,81), (81,101) ]

# Constants
file_name = __file__.split('/')[-1].split('.')[0]
plot_dir = f"{PLOT_DIR}/{file_name}"
if True:
    # Load data
    dataset, (user_summaries, text_embeddings, image_embeddings) = load_all_data(force=False)
    user_df = prefilter_data(user_summaries)

    # Load trajectories
    text_trajectories = load_text_trajectories(dataset, user_summaries, text_embeddings, only_pos=True)

    # Prompt-distribution
    trajectory_distances_text = get_trajectory_distances(dataset, text_trajectories, text_embeddings)

    # Get target image groupings by category
    target_meta = [user_df[user_df.target==t].iloc[0] for t in user_df.target.unique()]
    labels = defaultdict(list)
    for t in target_meta:
        for label_k, label_v in eval(t.target_labels).items():
            if label_v:
                labels[label_k].append(t.target)

# Now per user + across all users in a target, do clustering using "silhouette score" to auto-tune
#   the number of clusters
def autofind_clusters(X, max_n_clusters=None):
    if len(X) < 2:
        return {
            'model': None,
            'labels': [0],
            'N':len(X),
            'n_cluster': 1,
            'silhouette_score': -1
        }
    if max_n_clusters is None:
        max_n_clusters = len(X) // 2
    max_n_clusters = min(max_n_clusters, len(X) // 2)
    sil_score_max = -1 #this is the minimum possible score
    best_labels = [0]
    best_model = None
    best_n_clusters = 1
    for n_clusters in range(2,max_n_clusters+1):
        model = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=100, n_init=1)
        labels = model.fit_predict(X)
        if len(np.unique(labels)) == 1:
            continue
        sil_score = silhouette_score(X, labels)

        if sil_score > sil_score_max:
            sil_score_max = sil_score
            best_n_clusters = n_clusters
            best_labels = labels
            best_model = model
    return {
        'model': best_model,
        'labels':best_labels,
        'N':len(X),
        'n_cluster':best_n_clusters,
        'silhouette_score':sil_score_max
    }

def count_consecutive(l):
    last = l[0]
    same = [0]
    for l_i in l:
        if last == l_i:
            same[-1] += 1
        else:
            same.append(1)
            last = l_i
    same = np.array(same)
    return same
def _local_steerability(scores):
    if len(scores) <= 1:
        return np.nan
    return np.mean(np.sign(scores[1:]-scores[:-1]) > 0).astype(float)
def local_steerability(scores, clusters, increase=True, score_weighting=False, score_bin=None):
    if len(scores) <= 1:
        return np.nan

    if increase:
        local_changes = (np.sign(scores[1:]-scores[:-1]) > 0).astype(float)
        if score_weighting:
            local_changes *= (scores[:-1]/100)
    else:
        local_changes = (np.sign(scores[1:]-scores[:-1]) < 0).astype(float)
        if score_weighting:
            local_changes *= (1-scores[:-1]/100)
    if score_bin is not None:
        local_changes = local_changes[np.where((score_bin[0] <= scores[:-1]) & (scores[:-1] < score_bin[1]))[0]]
    if len(local_changes) == 0:
        return np.nan
    return np.mean(local_changes)

def global_steerability(scores, clusters, increase=True, score_bin=None):
    if len(clusters) <= 1 or len(np.unique(clusters)) <= 1:
        return np.nan
    idx = [0]+list(np.where(clusters[1:] != clusters[:-1])[0]+1)+[len(clusters)]
    max_in_cluster_area = np.array([
        np.max(scores[i1:i2])
        for i1,i2 in zip(idx[:-1], idx[1:])
    ])
    if score_bin is not None:
        max_in_cluster_area = max_in_cluster_area[np.where((score_bin[0] <= max_in_cluster_area[:-1]) & (max_in_cluster_area[:-1] < score_bin[1]))[0]]
    if len(max_in_cluster_area) <= 1:
        return np.nan
    return _local_steerability(max_in_cluster_area*(1 if increase else -1))

### vv -- Computing steerability -- vv
def score_to_bin(x, bins=BINS):
    for i,bin in enumerate(bins):
        if bin[0] <= x < bin[1]:
            return i
    return -1
def bin_to_text(idx, bins=BINS):
    if idx == -1:
        return "Start"
    return f"[{bins[idx][0]} -- {bins[idx][1]}]"

def gen_markov_model(use_val_data=False, use_rating=False, use_models=['SDv2.1'], normalization=1, bins=BINS):
    markov_models = {}
    n_per_target = {}
    for k,v in data_by_target.items():
        group_markov_model = defaultdict(list)
        n_per_target[k] = 0
        for target_i in v:
            for user_j in target_i:
                model = user_j['model_used']
                data_split = user_j['data_split']
                if (use_val_data) ^ (data_split == 'ArtWhisperer-Validation'):
                    continue
                if not np.any([mi==model for mi in use_models]):
                    continue
                if use_rating:
                    group_markov_model[-1].append(score_to_bin((100/9.5)*(user_j['rating_trajectory'][0]-0.5)))
                    if user_j['N'] < 2:
                        continue
                    for score_1,score_2 in zip(user_j['rating_trajectory'][:-1], user_j['rating_trajectory'][1:]):
                        group_markov_model[score_to_bin((100/9.5) * (score_1-0.5))].append(score_to_bin((100/9.5) * (score_2-0.5)))
                else:
                    group_markov_model[-1].append(score_to_bin(user_j['score_trajectory'][0]))
                    if user_j['N'] < 2:
                        continue
                    for score_1,score_2 in zip(user_j['score_trajectory'][:-1], user_j['score_trajectory'][1:]):
                        group_markov_model[score_to_bin(score_1)].append(score_to_bin(score_2))
                n_per_target[k] = n_per_target[k] + 1
        if len(group_markov_model) == 0:
            continue
        group_markov_model_probs = {}
        for bin in range(-1, len(bins)):
            if bin in group_markov_model:
                to_bins = group_markov_model[bin]
            else:
                to_bins = group_markov_model[-1]
            u,c = np.unique(to_bins, return_counts=True)
            max_u = len(bins)
            zip_d = {ci:normalization/(np.sum(c) + normalization*len(bins)) for ci in range(max_u)}
            zip_d.update({ui:(normalization+ci)/(np.sum(c) + normalization*len(bins)) for ui,ci in zip(u,c)})
            group_markov_model_probs[bin] = np.array(list([zip_d[ci] for ci in range(max_u)]))
        markov_models[k] = group_markov_model_probs
    return markov_models
def sim_markov_model(mmodel, seed=0, n_run=100, n_trial=100, stopping_time=False, bins=BINS):
    rng = np.random.RandomState(seed)
    trial_outcomes = []
    for _ in range(n_trial):
        start = -1
        i = -1
        for i in range(n_run):
            start = max(start, rng.choice(len(mmodel[start]), p=mmodel[start]))
            if stopping_time and (start == len(bins)-1):
                break
        if stopping_time:
            trial_outcomes.append(i+1)
        else:
            trial_outcomes.append(start)
    return trial_outcomes

def sim_and_plot(markov_models, plot=True, filename=None, bins=BINS):
    markov_sims = {}
    for mmodel_k, mmodel in markov_models.items():
        try:
            markov_sims[mmodel_k] = sim_markov_model(mmodel, stopping_time=True, bins=bins)
        except:
            continue

    group_avgs = defaultdict(list)
    for k,v in labels.items():
        for vi in v:
            if vi in markov_sims:
                group_avgs[k].append(np.mean(markov_sims[vi]))

    if plot:
        fig,ax = new_fig(nrows=1,ncols=1,figsize=(12,6))
        fig.set_tight_layout(True)
        for i,(k,v) in enumerate(group_avgs.items()):
            ax.bar(i, np.mean(v), 0.7,
                yerr=np.std(v, ddof=1)/np.sqrt(len(v)),
                capsize=4,
                label=f"{k}",
                edgecolor='black',
                color=PLOT_COLORS[i],
            )
            
        ax.yaxis.grid(True)
        x = np.arange(len(group_avgs))
        k_order = list(group_avgs.keys())
        ax.set_xticks(x)
        ax.set_xticklabels(k_order, rotation=45, fontsize=12)
        ax.set_ylabel("Expected stopping time")

        save_fig(filename=filename,
                    path=plot_dir, 
                    exts=['jpg'],
                    fig=fig,
                    tight=False)
        
        filepath = f"{plot_dir}/{filename}.jpg"
        save_with_cropped_whitespace(filepath)

    return markov_sims, group_avgs
### ^^ -- Computing steerability -- ^^

def figure_6():
    eps = 1e0
    mm_game = gen_markov_model(use_val_data=False, normalization=eps, bins=BINS)
    sim_and_plot(mm_game, plot=True, filename="Figure 6", bins=BINS)

def figure_10():
    eps = 1e0
    mm_s2_1 = gen_markov_model(use_val_data=True, use_models=[2.1], normalization=eps, bins=BINS)
    mm_s1_5 = gen_markov_model(use_val_data=True, use_models=[1.5], normalization=eps, bins=BINS)
    _, group_avgs_2_1 = sim_and_plot(mm_s2_1, plot=False, bins=BINS)
    _, group_avgs_1_5 = sim_and_plot(mm_s1_5, plot=False, bins=BINS)

    fig,ax = new_fig(nrows=1,ncols=1,figsize=(12,8))
    fig.set_tight_layout(True)
    bar_width = 0.4
    for j,group_avgs in enumerate([group_avgs_2_1, group_avgs_1_5]):
        for i,(k,v) in enumerate(group_avgs.items()):
            ax.bar(i + j*bar_width - (bar_width/2) + 1e-2*j, np.mean(v), bar_width,
                yerr=np.std(v, ddof=1)/np.sqrt(len(v)),
                capsize=4,
                label=f"{k}",
                hatch="/" if j == 1 else "",
                edgecolor='black',
                color=PLOT_COLORS[i],
            )
        
    ax.yaxis.grid(True)
    x = np.arange(len(group_avgs))
    k_order = list(group_avgs.keys())
    ax.set_xticks(x)
    ax.set_xticklabels(k_order, rotation=60, fontsize=10)
    ax.set_ylabel("Expected stopping time")

    circ1 = mpatches.Patch( facecolor="lightgray", alpha=1, hatch='',label='Model 1')
    circ2= mpatches.Patch( facecolor="lightgray", alpha=1, hatch='/',label='Model 2')
    ax.legend(handles = [circ1,circ2],loc=2)
    ax.set_ylim((0,14.5))

    filename="Figure 10"
    save_fig(
        filename=filename,
        path=plot_dir, 
        exts=['jpg'],
        fig=fig,
        tight=False
    )
    filepath = f"{plot_dir}/{filename}.jpg"
    save_with_cropped_whitespace(filepath)

def figure_13():
    eps = 1e0
    mm_s2_1 = gen_markov_model(use_val_data=True, use_models=[2.1], normalization=eps, bins=BINS)
    mm_s2_1r = gen_markov_model(use_val_data=True, use_rating=True, use_models=[1.5], normalization=eps, bins=BINS)
    _, group_avgs_2_1 = sim_and_plot(mm_s2_1, plot=False, bins=BINS)
    _, group_avgs_2_1r = sim_and_plot(mm_s2_1r, plot=False, bins=BINS)

    fig,ax = new_fig(nrows=1,ncols=1,figsize=(12,8))
    fig.set_tight_layout(True)
    bar_width = 0.4
    for j,group_avgs in enumerate([group_avgs_2_1, group_avgs_2_1r]):
        for i,(k,v) in enumerate(group_avgs.items()):
            ax.bar(i + j*bar_width - (bar_width/2) + 1e-2*j, np.mean(v), bar_width,
                yerr=np.std(v, ddof=1)/np.sqrt(len(v)),
                capsize=4,
                label=f"{k}",
                hatch="/" if j == 1 else "",
                edgecolor='black',
                color=PLOT_COLORS[i],
                )
        
    ax.yaxis.grid(True)
    x = np.arange(len(group_avgs))
    k_order = list(group_avgs.keys())
    ax.set_xticks(x)
    ax.set_xticklabels([ki.replace("Contains ", "").replace("Is ", "").replace(" content", "").capitalize().replace("Ai", "AI")
                        for ki in k_order], rotation=60, fontsize=10)
    ax.set_ylabel("Expected stopping time")

    circ1 = mpatches.Patch( facecolor="lightgray", alpha=1, hatch='',label='Score')
    circ2= mpatches.Patch( facecolor="lightgray", alpha=1, hatch='/',label='Human Rating')
    ax.legend(handles = [circ1,circ2],loc=2)
    ax.set_ylim((0,14.5))

    filename="Figure 13"
    save_fig(
        filename=filename,
        path=plot_dir, 
        exts=['jpg'],
        fig=fig,
        tight=False
    )
    filepath = f"{plot_dir}/{filename}.jpg"
    save_with_cropped_whitespace(filepath)


if __name__ == '__main__':
    # Using the score trajectories, not embeddings here
    cdata = text_trajectories['CLIP']
    data_by_target = {k: [cdata[k]] for k in cdata if len(cdata[k]) >= 10}
    if False:
        figure_6()
    if False:
        figure_10()
    if False:
        figure_13()