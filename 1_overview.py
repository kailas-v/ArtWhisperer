# -- High-level statistics etc. --
# Imports
from src.load_data import load_all_data, prefilter_data
from src.constants import PLOT_DIR
from src.plot_utils import save_fig, new_fig, setup_axes, PLOT_COLORS, save_with_cropped_whitespace, save_with_colored_rectangle
from src.im_utils import load_image
# from target_data_labels_auto import post_process_classifications, prompts
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import os
import re
import datetime
from collections import defaultdict, Counter

# Constants
file_name = __file__.split('/')[-1].split('.')[0]
plot_dir = f"{PLOT_DIR}/{file_name}"
# Load data
if True:
    dataset, (user_summaries, text_embeddings, image_embeddings) = load_all_data(force=False)
    user_df_ = prefilter_data(user_summaries)


# -- FIGURE 3 --
def figure_3():
    user_df = user_df_
    fig,axs = new_fig(nrows=1,ncols=2,figsize=(16,6))
    fig.subplots_adjust(wspace=0.35)
    queries_per_target = np.array(user_df.groupby(['user', 'target']).apply(lambda x: len(x)))
    min_, max_ = np.min(queries_per_target), np.max(queries_per_target)

    ax = axs[0]
    ax.hist(
        queries_per_target, 
        bins=np.logspace(np.log10(min_), np.log10(max_), num=20).astype(int),
        label="# Queries by User per Target",
        color=PLOT_COLORS[0],
        density=True,
        alpha=0.6
    )
    setup_axes([ax], no_legend=True)
    ax.set_xlabel("# Queries submitted per target")
    ax.set_ylabel("Density")
    ax.set_xscale('log')


    ax = axs[1]
    pos_prompt = list(user_df['pos_prompt'])
    neg_prompt = list(user_df['neg_prompt'])
    n_words_fn = lambda x: 0 if not (x) else len(re.findall(r'\b\w+\b', str(x)))
    words_per_query = [n_words_fn(ti) for ti in pos_prompt]
    words_per_query_neg = [n_words_fn(ti) for ti in neg_prompt]
    min_, max_ = np.min(queries_per_target), np.max(queries_per_target)
    ax.hist(
        words_per_query, 
        bins=np.logspace(np.log10(min_), np.log10(max_), num=20).astype(int),
        label="Positive Prompt",
        color=PLOT_COLORS[0],
        density=True,
        alpha=0.6
    )
    ax.hist(
        words_per_query_neg, 
        bins=np.logspace(np.log10(min_), np.log10(max_), num=20).astype(int),
        label="Negative Prompt",
        color=PLOT_COLORS[1],
        density=True,
        alpha=0.6
    )
    setup_axes([ax])
    ax.set_xlabel("# Words in query")
    ax.set_ylabel("Density")
    ax.set_xscale('log')

    print(f"Average queries_per_target: {np.mean(queries_per_target):0.2f}")
    print(f"Average words_per_query: {np.mean(words_per_query):0.2f}")
    print(f"Average words_per_query_neg: {np.mean(words_per_query_neg):0.2f}")

    save_fig(filename=f"Figure 3", 
            path=plot_dir,
            show=False, 
            fig=fig)

# -- FIGURE generation for Figures 2, 8 --
def figure_28_pre():
    user_df = user_df_
    plot_dir_ = f"{plot_dir}/prompt-progression-examples"
    os.makedirs(plot_dir_, exist_ok=True)

    target_images = ['435f38f3084d9e1e','eef65c3de840b914','0a726071379647b8','333289ce63f49e55','dd37283fa547d826','ebfd65ac1e8ae3b3', '1857fff409c62abc','0843b4b67f7e504f','0acd66c9a234ae02','7d8b37dbb8df1a4c']
    
    n_select = 4
    n_user = 4
    for target_image in target_images:
        # Just looking at entries with this target image
        df = user_df[user_df.target==target_image]

        # Get an entry in the dataset that contains the target image and target meta info
        target_meta = dataset[int(df.iloc[0].map_to_dataset)]
        target_prompt = target_meta['target_positive_prompt']

        max_attempts = df.groupby('user').apply(lambda x: len(x)).max()
        scores = []
        for _,dfg in df.groupby('user'):
            scores.append(np.zeros(max_attempts))
            scores[-1][:len(dfg)] = dfg.score.cummax()
            scores[-1][len(dfg):] = np.max(dfg.score)
        scores = np.array(scores)
        mean_scores = np.mean(scores, axis=0)
        quantile_scores = np.quantile(scores, q=[0.25,0.75], axis=0)

        best_score = list(df.groupby('user').apply(lambda x: x.improved_score.sum()).to_dict().items())
        best_score = sorted(best_score, key=lambda x: -x[1])
        best_score = [(k,v) for k,v in best_score]

        user_list = [k for k,_ in best_score]
        counter = n_user
        for i in range(len(user_list)):
            df_i = df[df.user==user_list[i]].sort_values(['trajectory_index'])
            user_scores = np.array(df_i.score)
            user_prompts = np.array(df_i.pos_prompt)
            if df_i.score.max() < 80:
                continue
            where_improved = np.where(np.array(df_i.improved_score))[0]
            map_to_datasets = np.array(df_i.map_to_dataset)[where_improved]
            if len(map_to_datasets) == 0:
                continue
            select_idx = np.linspace(0, len(map_to_datasets)-1, num=n_select).astype(int)
            images = map_to_datasets[select_idx]
            user_scores_x = where_improved[select_idx]
            user_scores_shown = user_scores[user_scores_x]
            user_prompts_shown = user_prompts[user_scores_x]
            imgs = []
            for image in images:
                imgs.append(dataset[int(image)]['generated_image'])
            if len(imgs) != n_select:
                continue
            imgs.append(target_meta['target_image'])

            # Show image progression and target image
            fig = plt.figure(figsize=(24, 18))
            grid = gridspec.GridSpec(1, 3, width_ratios=[4.15, 1, 1], hspace=0, wspace=0.25, top=1, bottom=0.3, left=0, right=1)
            gs1 = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=grid[0], wspace=0.05)
            gs2 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=grid[1], wspace=0)
            gs3 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=grid[2], wspace=0)

            txts = []
            for im_i in range(len(imgs)):
                if (im_i < 4) and (len(user_prompts_shown) <= im_i):
                    continue
                if im_i < 4:
                    ax = plt.subplot(gs1[im_i])
                else:
                    ax = plt.subplot(gs2[im_i-4])
                ax.imshow(imgs[im_i])
                ax.axis('off')
                if im_i < 4:
                    p = user_prompts_shown[im_i].replace("，", ",")
                else:
                    p = target_prompt

                if im_i < n_select:
                    txt = ax.text(0.5, -0.025, f"$\\mathbf{{Score: {user_scores_shown[im_i]}}}$",
                                ha='center', va='top', transform=ax.transAxes, 
                                fontfamily='monospace', fontsize=14, wrap=True)
                    txt2 = ax.text(0.5, -0.10, p,
                                    ha='center', va='top', transform=ax.transAxes, 
                                    fontfamily='monospace', fontsize=14, wrap=True)
                    txt2._get_wrap_line_width = lambda : 275
                    txts.append((txt, txt2))
                else:
                    txt = None
                    txt2 = ax.text(0.5, -0.025, p,
                                    ha='center', va='top', transform=ax.transAxes, 
                                    fontfamily='monospace', fontsize=14, wrap=True)
                    txt2._get_wrap_line_width = lambda : 275
                    txts.append((txt, txt2))
            
            # Show trajectory
            x_ = np.arange(len(user_scores))
            ax = plt.subplot(gs3[-1])
            ax.plot(
                x_+1,
                mean_scores[:len(x_)],
                label="Average score after # prompts",
                linestyle='--',
                linewidth=8,
                color='blue',
            )
            ax.fill_between(
                x_+1,
                quantile_scores[0,:len(x_)],
                quantile_scores[1,:len(x_)],
                color='blue',
                alpha=0.2
            )
            ax.plot(
                x_+1,
                user_scores,
                label="User score after # prompts",
                linestyle='--',
                linewidth=4,
                color='red'
            )
            ax.scatter(
                user_scores_x+1,
                user_scores_shown,
                label="Displayed images",
                marker='o',
                s=196,
                color='orange',
                edgecolors='black',
            )
            ax.tick_params(axis='both', which='major', labelsize=14)
            ax.tick_params(axis='both', which='minor', labelsize=14)
            ax.set_ylabel("Score", fontsize=18)
            ax.set_xlabel("# Interactions", fontsize=18)
            ax.set_ylim((0-5,100+5))
            if len(x_) <= 0:
                x = list(range(1,len(x_)+1,2))
                ax.set_xticks(x)
            ax.grid()
            asp = np.diff(ax.get_xlim())[0] / np.diff(ax.get_ylim())[0]
            ax.set_aspect(asp)

            # Save
            filename = f"user_image--target-{target_image}_user-{i+1}"
            save_fig(filename=filename,
                     path=plot_dir_, 
                     exts=['jpg'],
                     fig=fig,
                     tight=False)
            
            
            
            filepath = f"{plot_dir_}/{filename}.jpg"
            save_with_cropped_whitespace(filepath)
            
            counter -= 1
            if counter == 0:
                break

# -- FIGURE 2 --
def figure_2():
    selected_for_paper = [
        ('1857fff409c62abc', 4),
        ('0843b4b67f7e504f', 5),
        ('0acd66c9a234ae02', 2),
        ('7d8b37dbb8df1a4c', 3)
    ]
    
    filepath = lambda x: f"{plot_dir}/prompt-progression-examples/user_image--target-{x[0]}_user-{x[1]}.jpg"
    imgs = [load_image(filepath(sfpi), as_PIL=True) for sfpi in selected_for_paper]
    heights = [img.size[1] for img in imgs]
    min_height = min(heights)
    height_ratios = [hi/min_height for hi in heights]

    fig = plt.figure(figsize=(24, 6*len(selected_for_paper)))
    grid = gridspec.GridSpec(len(selected_for_paper), 1, height_ratios=height_ratios, hspace=0.1)
    for g,im in zip(grid,imgs):
        ax = plt.subplot(g)
        ax.imshow(im)
        ax.axis('off')

    filename = f"Figure 2"
    save_fig(filename=filename,
            path=plot_dir, 
            exts=['jpg'],
            fig=fig,
            tight=False)
    save_with_cropped_whitespace(f"{plot_dir}/{filename}.jpg")
    IMG_HEIGHT = 245
    skip_rect_h0 = [11, 540, 1033, 1497]
    skip_rects=[
        [[sri, sri+IMG_HEIGHT], [1142, 1386]] for sri in skip_rect_h0
    ]
    save_with_colored_rectangle(f"{plot_dir}/{filename}.jpg", rect=[[0, None], [1132, 1396]], all=True, skip_rects=skip_rects, color=(0,255,0), alpha=0.2)

# -- FIGURE 8, APPENDIX --
def figure_8_appendix():
    selected_for_paper = [
        ('435f38f3084d9e1e', 5),
        ('eef65c3de840b914', 1),
        ('0a726071379647b8', 1),
        ('333289ce63f49e55', 3),
        ('dd37283fa547d826', 2),
        ('ebfd65ac1e8ae3b3', 1)
    ]

    filepath = lambda x: f"{plot_dir}/prompt-progression-examples/user_image--target-{x[0]}_user-{x[1]}.jpg"
    imgs = [load_image(filepath(sfpi), as_PIL=True) for sfpi in selected_for_paper]
    heights = [img.size[1] for img in imgs]
    min_height = min(heights)
    height_ratios = [hi/min_height for hi in heights]

    fig = plt.figure(figsize=(24, 6*len(selected_for_paper)))
    grid = gridspec.GridSpec(len(selected_for_paper), 1, height_ratios=height_ratios, hspace=0.1)
    for g,im in zip(grid,imgs):
        ax = plt.subplot(g)
        ax.imshow(im)
        ax.axis('off')

    filename = f"Figure 8 (Appendix)"
    save_fig(filename=filename,
            path=plot_dir, 
            exts=['jpg'],
            fig=fig,
            tight=False)
    save_with_cropped_whitespace(f"{plot_dir}/{filename}.jpg")
    IMG_HEIGHT = 225
    skip_rect_h0 = [11, 475, 917, 1380, 1870, 2319]
    skip_rects=[
        [[sri, sri+IMG_HEIGHT], [1057, 1284]] for sri in skip_rect_h0
    ]
    save_with_colored_rectangle(f"{plot_dir}/{filename}.jpg", rect=[[0, None], [1047, 1294]], all=True, skip_rects=skip_rects, color=(0,255,0), alpha=0.2)

# -- TABLE = # users, # targets, avg. # queries per target
def table_1(train):
    user_df = user_df_
    if train:
        user_df = user_df[user_df.data_split.apply(lambda x: 'Validation' not in x)]
    n_user = len(user_df.user.unique())
    targets_selected = user_df.target.unique()
    n_target = len(user_df.target.unique())
    n_query = len(user_df)
    queries_per_ut = np.array(user_df.groupby(['user', 'target']).apply(lambda x: len(x)))
    avg_queries_per_target = n_query / n_target
    avg_queries_per_ut = np.mean(queries_per_ut)
    d = dict(
        n_user=n_user,
        n_target=n_target,
        n_queries=n_query,
        avg_queries_per_target=avg_queries_per_target,
        avg_queries_per_ut=avg_queries_per_ut,
    )

    target_meta = [
        user_df[user_df.target==t].iloc[0] for t in targets_selected
    ]
    labels = defaultdict(list)
    for t in target_meta:
        for label_k, label_v in eval(t.target_labels).items():
            if label_v:
                labels[label_k].append(t.target)
    table_start = r"""
\begin{table}[ht]
\caption{Dataset Overview. Each row contains summary data for a different subset of the dataset. Subsets may overlap.} 
\label{tbl:overview_of_data}
\vskip -0.3in
\begin{center}
\begin{tabular}{>{\centering\arraybackslash}m{1.2cm}|>{\centering\arraybackslash}m{1.1cm}|>{\centering\arraybackslash}m{1.1cm}|>{\centering\arraybackslash}m{1.3cm}|>{\centering\arraybackslash}m{1.1cm}|>{\centering\arraybackslash}m{1.2cm}|>{\centering\arraybackslash}m{4cm}}
\toprule
\small \# Players & \small \# Target Images & \small \# Interactions & \small Average \# Prompts & \small Average Score & \small Median Duration & \small Category \\ 
\hline"""
    table_end = r"""
\bottomrule
\end{tabular} 
\end{center}
\vskip -0.1in
\end{table}"""
    table_middle = []
    time_taken = list(user_df.time_taken)
    time_taken = [datetime.datetime.strptime(ti.split("days ")[-1], '%H:%M:%S') for ti in time_taken if ti]
    time_taken = [datetime.timedelta(hours=ti.hour, minutes=ti.minute, seconds=ti.second) for ti in time_taken]
    table_middle.append(
        f"\\hline\n\\small \\textbf{{{len(user_df.user.unique())}}} & \\small \\textbf{{{len(user_df.target.unique())}}} & \\small \\textbf{{{len(user_df)}}} & \\small \\textbf{{{user_df.groupby(['user','target']).apply(lambda x: len(x)).mean():0.2f}}} & \\small \\textbf{{{user_df.score.mean():0.2f}}} & \\small \\textbf{{{np.median(time_taken).seconds} s}} & \\small \\textbf{{{'Total'}}} \\\\\n\\hline\\hline"
    )
    for i,(label_name,targets_with_label) in enumerate(labels.items()):
        df = user_df[user_df.target.isin(targets_with_label)]
        time_taken = list(df.time_taken)
        time_taken = [datetime.datetime.strptime(ti.split("days ")[-1], '%H:%M:%S') for ti in df.time_taken if ti]
        time_taken = [datetime.timedelta(hours=ti.hour, minutes=ti.minute, seconds=ti.second) for ti in time_taken]
        hline = "\\hline"
        row_txt = f"\\small {len(df.user.unique())} & \\small {len(targets_with_label)} & \\small {len(df)} & \\small {df.groupby(['user','target']).apply(lambda x: len(x)).mean():0.2f} & \\small {df.score.mean():0.2f} & \\small {np.median(time_taken).seconds} s & \\small {label_name} \\\\\n{hline if i < len(labels)-1 else ''}"
        table_middle.append(row_txt)
    table = table_start + "\n".join(table_middle) + "\n" + table_end

    table = table.replace("manmade", "man-made")
    print(table)


if __name__ == '__main__':
    if False:
        figure_3()
    if False:
        figure_28_pre()
        figure_2()
        figure_8_appendix()
    if False:
        table_1(train=True)