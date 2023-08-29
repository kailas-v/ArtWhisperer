# -- Prompt diversity --
# Imports
from src.load_data import load_all_data, prefilter_data, load_text_trajectories, get_trajectory_distances
from src.constants import PLOT_DIR
from src.plot_utils import save_fig, new_fig, setup_axes, PLOT_COLORS, save_with_cropped_whitespace, save_with_colored_rectangle
from src.im_utils import load_image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import gridspec
from collections import defaultdict
import os
from scipy.stats import kstest, ttest_ind
from itertools import combinations
import umap
from PIL import Image, ImageOps


# Constants
file_name = __file__.split('/')[-1].split('.')[0]
plot_dir = f"{PLOT_DIR}/{file_name}"
os.makedirs(plot_dir, exist_ok=True)
if True:
    # Load data
    dataset, (user_summaries, text_embeddings, image_embeddings) = load_all_data(force=False)
    user_df = prefilter_data(user_summaries)

    # Load trajectories
    text_trajectories = load_text_trajectories(dataset, user_summaries, text_embeddings, only_pos=True)

    # Prompt-distribution
    trajectory_distances_text = get_trajectory_distances(dataset, text_trajectories, text_embeddings)



# -- FIGURE generation for Figures 4, 9 --
def figure_49_pre():
    plot_dir_ = f"{plot_dir}/diverse-prompt-examples"
    os.makedirs(plot_dir_, exist_ok=True)

    target_images = ['04d31d1012d9c82c', '13930dbd2774cf35', 'ee4ce65b8a1ed6e9', 'b45d006e147d2ec6', '34159ab10beaed5b', 'b2d5dd14d3252fa2']

    n_select = 4
    MIN_USER_SCORE = 90

    for target_image in target_images:
        df = user_df[user_df.target==target_image]
        target_meta = dataset[int(df.iloc[0].map_to_dataset)]
        target_embedding = np.array(target_meta['target_image_embedding']['value'])
        target_prompt = target_meta['target_positive_prompt']

        best_score = list(df.groupby('user').apply(lambda x: x.improved_score.sum()).to_dict().items())
        try:
            best_score = sorted(best_score, key=lambda x: -x[1])
        except:
            continue
        best_score = [(k,v) for k,v in best_score]

        user_list = [k for k,_ in best_score]
        imgs = []
        all_imgs = []
        all_imgs_txt = []
        for i in range(len(user_list)):
            df_i = df[df.user==user_list[i]].sort_values(['trajectory_index'])
            best_img = np.argmax(df_i.score)
            first_img = 0
            user_score = np.array(df_i.score)[best_img]
            scores = np.array(df_i.score)
            user_prompt = np.array(df_i.pos_prompt)[best_img]
            map_to_dataset = np.array(df_i.map_to_dataset)[best_img]
            img_meta = dataset[int(map_to_dataset)]
            embedding = np.array(img_meta['generated_image_embedding']['value'])

            first_user_score = np.array(df_i.score)[first_img]
            first_user_prompt = np.array(df_i.pos_prompt)[first_img]
            first_map_to_dataset = np.array(df_i.map_to_dataset)[first_img]
            first_img_meta = dataset[int(first_map_to_dataset)]
            first_embedding = np.array(first_img_meta['generated_image_embedding']['value'])
            
            text_embedding = text_embeddings[img_meta['generated_positive_prompt']]['CLIP']
            first_text_embedding = text_embeddings[first_img_meta['generated_positive_prompt']]['CLIP']

            if user_score >= MIN_USER_SCORE:
                imgs.append({
                    'img':img_meta['generated_image'],
                    'prompt':user_prompt,
                    'score':user_score,
                    'embedding':embedding,
                    'text_embedding':text_embedding,

                    'first_img':first_img_meta['generated_image'],
                    'first_user_prompt': first_user_prompt,
                    'first_user_score': first_user_score,
                    'first_embedding':first_embedding,
                    'first_text_embedding':first_text_embedding
                })
            for ei,map_to_dataset in enumerate(list(df_i.map_to_dataset)):
                if ((ei == first_img) or (ei == best_img)) and (user_score >= MIN_USER_SCORE):
                    continue
                if scores[ei] < 50:
                    continue
                meta = dataset[int(map_to_dataset)]
                if meta and 'generated_image_embedding' in meta:
                    embedding = np.array(img_meta['generated_image_embedding']['value'])
                    text_embedding = text_embeddings[img_meta['generated_positive_prompt']]['CLIP']
                    all_imgs.append(embedding)
                    all_imgs_txt.append(text_embedding)
            
        if len(imgs) < n_select:
            continue
        
        

        def find_furthest_vectors(vectors, k):
            n = len(vectors)
            if k >= n:
                return vectors, np.arange(len(vectors))
            vectors = np.array(vectors)

            # Calculate cosine similarity matrix
            similarity_matrix = np.dot(vectors, vectors.T) / (np.linalg.norm(vectors, axis=1) * np.linalg.norm(vectors, axis=1)[:, np.newaxis])

            max_distance = 1.0
            max_subset = None

            # Generate all combinations of k vectors
            rng = np.random.RandomState(0)
            max_n = 100
            vector_combinations = list(combinations(range(n), k))
            if max_n < len(vector_combinations):
                idxs = rng.choice(len(vector_combinations), size=max_n, replace=False)
                vector_combinations = [vector_combinations[idx] for idx in idxs]

            for combination in vector_combinations:
                # Calculate the pairwise cosine similarity for the current combination
                combination_similarity = similarity_matrix[np.ix_(combination, combination)]
                combination_similarity -= np.eye(len(combination_similarity))

                # Calculate the minimum similarity within the combination
                max_similarity = np.max(combination_similarity)

                if max_similarity < max_distance:
                    max_distance = max_similarity
                    max_subset = combination

            return [vectors[i] for i in max_subset], max_subset
        
        vectors_txt_ = [img['text_embedding'] for img in imgs]
        _, idxs = find_furthest_vectors(vectors_txt_, n_select)
        all_imgs.extend([img['embedding'] for idx,img in enumerate(imgs) if idx not in idxs])
        all_imgs.extend([img['first_embedding'] for idx,img in enumerate(imgs) if idx not in idxs])
        all_imgs_txt.extend([img['text_embedding'] for idx,img in enumerate(imgs) if idx not in idxs])
        all_imgs_txt.extend([img['first_text_embedding'] for idx,img in enumerate(imgs) if idx not in idxs])
        imgs = [img for idx,img in enumerate(imgs) if idx in idxs]
        vectors = [img['embedding'] for img in imgs]
        vectors_txt = [img['text_embedding'] for img in imgs]
        vectors_start = [img['first_embedding'] for img in imgs]
        vectors_txt_start = [img['first_text_embedding'] for img in imgs]
        imgs.append({'img':target_meta['target_image']})


        # Show image progression and target image
        fig = plt.figure(figsize=(24, 18))
        grid = gridspec.GridSpec(1, 3, width_ratios=[4.15, 1, 2.35], hspace=0, wspace=0.25, top=1, bottom=0.3, left=0, right=1)
        gs1 = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=grid[0], wspace=0.05)
        gs2 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=grid[1], wspace=0)
        gs3 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=grid[2], wspace=0.35)

        txts = []
        for im_i,img in enumerate(imgs):
            if im_i < 4:
                ax = plt.subplot(gs1[im_i])
            else:
                ax = plt.subplot(gs2[im_i-4])
            ax.imshow(img['img'])
            ax.axis('off')
            if im_i < 4:
                if 'prompt' not in img:
                    break
                p = img['prompt']
                if "real,real,real," in p:
                    p1,p2 = p.split("real,real,real,")
                    p = p1 + "\nreal,real,real," + p2
            else:
                p = target_prompt
            
            if im_i < n_select:
                txt = ax.text(0.5, -0.025, f"$\\mathbf{{Score: {img['score']}}}$",
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
        if len(txts) < 5:
            continue
            
        umap_embedding = umap.UMAP().fit_transform(all_imgs + vectors + vectors_start + [target_embedding])

        ax = plt.subplot(gs3[0])
        
        for i in range(len(all_imgs), len(umap_embedding)-len(vectors)-1):
            # Plot the points
            plot_kwargs = {
                'marker':'o',
                's':64,
                'edgecolor':'black',
                'color':PLOT_COLORS[2],
                'alpha':1
            }
            ax.scatter(*umap_embedding[i], **plot_kwargs)
            plot_kwargs = {
                'marker':'o',
                's':64,
                'edgecolor':'black',
                'color':PLOT_COLORS[3],
                'alpha':1
            }
            ax.scatter(*umap_embedding[i+len(vectors)], **plot_kwargs)

            # Add an arrow from point1 to point2
            arrow_kwargs = dict(arrowstyle='-|>, head_width=0.75', linewidth=2, color='black', alpha=0.75)
            ax.annotate('', xy=umap_embedding[i], xytext=umap_embedding[i+len(vectors)], arrowprops=arrow_kwargs)

        ax.scatter(
            umap_embedding[-1,0],
            umap_embedding[-1,1],
            marker='o',
            s=144,
            color=PLOT_COLORS[1],
            edgecolor='black',
            alpha=1
        )
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.tick_params(axis='both', which='minor', labelsize=14)
        ax.set_ylabel("UMAP-2 (Image)", fontsize=18)
        ax.set_xlabel("UMAP-1 (Image)", fontsize=18)
        ax.grid()
        asp = np.diff(ax.get_xlim())[0] / np.diff(ax.get_ylim())[0]
        ax.set_aspect(asp)

        vectors.append(target_embedding)

        umap_embedding = umap.UMAP(
            n_neighbors=15,
            min_dist=1e-1
        ).fit_transform(all_imgs_txt + vectors_txt + vectors_txt_start + [text_embeddings[target_prompt]['CLIP']])
        vectors_txt.append(text_embeddings[target_prompt]['CLIP'])
        
        ax = plt.subplot(gs3[1])
        for i in range(len(all_imgs), len(umap_embedding)-len(vectors)-1):
            # Plot the points
            plot_kwargs = {
                'marker':'o',
                's':64,
                'edgecolor':'black',
                'color':PLOT_COLORS[2],
                'alpha':1
            }
            ax.scatter(*umap_embedding[i], **plot_kwargs)
            plot_kwargs = {
                'marker':'o',
                's':64,
                'edgecolor':'black',
                'color':PLOT_COLORS[3],
                'alpha':1
            }
            ax.scatter(*umap_embedding[i+len(vectors)], **plot_kwargs)

            # Add an arrow from point1 to point2
            arrow_kwargs = dict(arrowstyle='-|>, head_width=0.75', linewidth=2, color='black', alpha=0.75)
            ax.annotate('', xy=umap_embedding[i], xytext=umap_embedding[i+len(vectors)], arrowprops=arrow_kwargs)

        ax.scatter(
            umap_embedding[-1,0],
            umap_embedding[-1,1],
            label="Target prompt",
            marker='o',
            s=144,
            color=PLOT_COLORS[1],
            edgecolor='black',
            alpha=1
        )
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.tick_params(axis='both', which='minor', labelsize=14)
        ax.set_ylabel("UMAP-2 (Prompt Text)", fontsize=18)
        ax.set_xlabel("UMAP-1 (Prompt Text)", fontsize=18)
        ax.grid()
        asp = np.diff(ax.get_xlim())[0] / np.diff(ax.get_ylim())[0]
        ax.set_aspect(asp)

        # Save
        filename = f"user_image--target-{target_image}"
        save_fig(filename=filename,
                    path=plot_dir_, 
                    exts=['jpg'],
                    fig=fig,
                    tight=False)
        
        filepath = f"{plot_dir_}/{filename}.jpg"
        print(filepath)
        save_with_cropped_whitespace(filepath)

# -- FIGURE 3 --
def figure_4():
    selected_for_paper = ['04d31d1012d9c82c', '13930dbd2774cf35', 'ee4ce65b8a1ed6e9']

    filepath = lambda x: f"{plot_dir}/diverse-prompt-examples/user_image--target-{x}.jpg"
    imgs = [load_image(filepath(sfpi), as_PIL=True) for sfpi in selected_for_paper]

    fig = plt.figure(figsize=(6*len(selected_for_paper), 16))
    grid = gridspec.GridSpec(len(selected_for_paper), 1, hspace=0.)
    for g,im in zip(grid,imgs):
        ax = plt.subplot(g)
        print(im.size)
        ax.imshow(im)
        ax.axis('off')

    filename = f"Figure 4"
    save_fig(filename=filename,
            path=plot_dir, 
            exts=['jpg'],
            fig=fig,
            tight=False)
    save_with_cropped_whitespace(f"{plot_dir}/{filename}.jpg")
    IMG_HEIGHT = 160
    skip_rect_h0 = [11, 428, 850]
    skip_rects=[
        [[sri, sri+IMG_HEIGHT], [762, 923]] for sri in skip_rect_h0
    ]
    save_with_colored_rectangle(f"{plot_dir}/{filename}.jpg", rect=[[0, None], [752, 933]], all=True, skip_rects=skip_rects, color=(0,255,0), alpha=0.2)

# -- FIGURE 9, APPENDIX --
def figure_9_appendix():
    selected_for_paper = ['b45d006e147d2ec6', '34159ab10beaed5b', 'b2d5dd14d3252fa2']

    filepath = lambda x: f"{plot_dir}/diverse-prompt-examples/user_image--target-{x}.jpg"
    imgs = [load_image(filepath(sfpi), as_PIL=True) for sfpi in selected_for_paper]

    fig = plt.figure(figsize=(6*len(selected_for_paper), 16))
    grid = gridspec.GridSpec(len(selected_for_paper), 1, hspace=0.)
    for g,im in zip(grid,imgs):
        ax = plt.subplot(g)
        print(im.size)
        ax.imshow(im)
        ax.axis('off')

    filename = f"Figure 9 (Appendix)"
    save_fig(filename=filename,
            path=plot_dir, 
            exts=['jpg'],
            fig=fig,
            tight=False)
    save_with_cropped_whitespace(f"{plot_dir}/{filename}.jpg")
    IMG_HEIGHT = 160
    skip_rect_h0 = [15, 428, 843]
    skip_rects=[
        [[sri, sri+IMG_HEIGHT], [762, 923]] for sri in skip_rect_h0
    ]
    save_with_colored_rectangle(f"{plot_dir}/{filename}.jpg", rect=[[0, None], [752, 933]], all=True, skip_rects=skip_rects, color=(0,255,0), alpha=0.2)

    
# -- FIGURE 5 --
def figure_5():
    fig = plt.figure(figsize=(12, 4))
    grid = gridspec.GridSpec(1, 3, wspace=0.45)
    subplot1 = plt.subplot(grid[0])
    subplot2 = plt.subplot(grid[1])
    subplot3 = plt.subplot(grid[2])


    text_trajectories_by_user = defaultdict(lambda: defaultdict(list))
    avg_by_target = defaultdict(list)
    for target, trajectories in text_trajectories['CLIP'].items():
        for ti in trajectories:
            traj = ti['trajectory']
            text_trajectories_by_user[ti['user']][target].append(traj)
            avg_by_target[target].append(traj)
    for k,v in avg_by_target.items():
        avg_by_target[k] = np.mean(np.concatenate(v, axis=0), axis=0)
    for k,v in text_trajectories_by_user.items():
        for kk,vv in v.items():
            v[kk] = np.concatenate(vv, axis=0)
        
    avg_embed_by_user = defaultdict(dict)
    avg_embed_by_target = defaultdict(list)
    for user, trajectory in text_trajectories_by_user.items():
        for target, traj in trajectory.items():
            t_avg_embed = np.mean(traj, axis=0)
            avg_embed_by_user[user][target] = t_avg_embed
            avg_embed_by_target[target].append(t_avg_embed)

    rng = np.random.RandomState(0)
    real_users = []
    fake_users = []
    for user, ts in avg_embed_by_user.items():
        if len(ts) < 2: continue
        targets = list(ts.keys())
        # real
        x = np.concatenate([ti[None,:] for ti in ts.values()], axis=0)
        x_u = np.mean(x, axis=0)
        real_users.append(np.sqrt(np.mean(np.linalg.norm(x-x_u, axis=1)**2)))

        fake_x = [avg_embed_by_target[t][rng.randint(len(avg_embed_by_target[t]))] for t in targets]
        fake_x = np.concatenate([ti[None,:] for ti in fake_x])
        fake_x_u = np.mean(fake_x, axis=0)
        fake_users.append(np.sqrt(np.mean(np.linalg.norm(fake_x-fake_x_u, axis=1)**2)))

    ax = subplot3
    ax.hist(
        real_users, 
        bins=10,
        label="Real users",
        color=PLOT_COLORS[0],
        density=True,
        alpha=0.5
    )

    ax.hist(
        fake_users, 
        bins=10,
        label="Permuted users",
        color=PLOT_COLORS[1],
        density=True,
        alpha=0.5
    )

    KS_TEST = kstest(
        real_users,
        fake_users
    )
    print(KS_TEST)
    T_TEST = ttest_ind(
        real_users,
        fake_users,
        equal_var=False
    )
    print(T_TEST)
    print(f"MEAN real_users = {np.mean(real_users)}")
    print(f"MEAN permuted_users = {np.mean(fake_users)}")

    setup_axes([ax])
    ax.legend()
    ax.set_xlabel("Across-prompt embedding standard deviation")
    ax.set_ylabel("Density")

    print("\n" + "="*40 + "\n")



    scores_f = [xi for t in trajectory_distances_text['CLIP'].keys() for xi in trajectory_distances_text['CLIP'][t]['scores_f']]
    scores_b = [xi for t in trajectory_distances_text['CLIP'].keys() for xi in trajectory_distances_text['CLIP'][t]['scores_b']]

    print(f"Average score start: {np.mean(scores_f)}")
    print(f"Average score best: {np.mean(scores_b)}")

    for et,a in trajectory_distances_text.items():
        if et != "CLIP": continue

        ax = subplot1
        ax.hist(
            [data_i for target,data in a.items() for data_i in data['dist_from_avg_f']], 
            bins=10,
            label="First prompt",
            color=PLOT_COLORS[0],
            density=True,
            alpha=0.5
        )
        ax.hist(
            [data_i for target,data in a.items() for data_i in data['dist_from_avg_b']], 
            bins=10,
            label="Best prompt",
            color=PLOT_COLORS[1],
            density=True,
            alpha=0.5
        )

        KS_TEST = kstest(
            [data_i for target,data in a.items() for data_i in data['dist_from_avg_f']],
            [data_i for target,data in a.items() for data_i in data['dist_from_avg_b']]
        )
        print(KS_TEST)
        T_TEST = ttest_ind(
            [data_i for target,data in a.items() for data_i in data['dist_from_avg_f']],
            [data_i for target,data in a.items() for data_i in data['dist_from_avg_b']],
            equal_var=False
        )
        print(T_TEST)
        print(f"MEAN dist_from_avg_f = {np.mean([data_i for target,data in a.items() for data_i in data['dist_from_avg_f']])}")
        print(f"MEAN dist_from_avg_b = {np.mean([data_i for target,data in a.items() for data_i in data['dist_from_avg_b']])}")

        setup_axes([ax])
        ax.set_xlabel("User prompt embedding standard deviation")
        ax.set_ylabel("Density")

        print("\n" + "="*40 + "\n")


        ax = subplot2
        ax.hist(
            [data_i for target,data in a.items() for data_i in data['dist_from_avg']], 
            bins=10,
            label="Real users",
            color=PLOT_COLORS[0],
            density=True,
            alpha=0.5
        )
        ax.hist(
            [data_i for target,data in a.items() for data_i in data['fake_dist_from_avg']], 
            bins=10,
            label="Permuted users",
            color=PLOT_COLORS[1],
            density=True,
            alpha=0.5
        )

        KS_TEST = kstest(
            [data_i for target,data in a.items() for data_i in data['dist_from_avg']],
            [data_i for target,data in a.items() for data_i in data['fake_dist_from_avg']]
        )
        print(KS_TEST)
        T_TEST = ttest_ind(
            [data_i for target,data in a.items() for data_i in data['dist_from_avg']],
            [data_i for target,data in a.items() for data_i in data['fake_dist_from_avg']],
            equal_var=False
        )
        print(T_TEST)
        print(f"MEAN dist_from_avg      = {np.mean([data_i for target,data in a.items() for data_i in data['dist_from_avg']])}")
        print(f"MEAN fake_dist_from_avg = {np.mean([data_i for target,data in a.items() for data_i in data['fake_dist_from_avg']])}")

        
        setup_axes([ax])
        ax.set_xlabel("User prompt embedding standard deviation")
        ax.set_ylabel("Density")
        
        break

    save_fig(filename=f"Figure 5", 
            path=plot_dir,
            show=False, 
            fig=fig)

steerability_scores = {'6448adc8cb7ba64e': 1.792, 'b9b1c233f4e8a632': 1.902, '78d95d7bc56a9a1c': 1.954, '08cdb3012619c11d': 2.04, 'ed5b6018fef927f3': 2.055, 'ab091e0e5bc73e45': 2.319, '0a726071379647b8': 2.539, 'd892dd4f17fa7a4c': 2.562, '748e63c8ce2867ec': 2.669, 'f84b69b5805012c3': 2.672, 'd8291c3449d29a02': 2.846, '91d8151c56553d38': 2.86, '4a27fd8e8c8588e0': 3.274, 'b1db76776a319a1b': 3.289, '2774eadc4de8dcc3': 3.304, '7f0d374975856024': 3.418, 'e03b4bdfa4b6528a': 3.429, 'c90aa96d17781fbc': 3.646, 'f9521df071f92a1d': 3.81, '99ac23744c7414ca': 4.138, 'eea396e43a48d922': 4.212, 'ccac0496521f5b9d': 4.262, '13930dbd2774cf35': 4.299, '96131b48a4bc3d8d': 4.518, '047092a77b422647': 4.607, 'be0d0357db81f723': 4.62, 'eef65c3de840b914': 4.642, '1857fff409c62abc': 4.656, 'e5da465f69b372f6': 4.876, 'd10cf286ba102b2d': 4.914, '270ae531dbb5da1e': 4.926, '0acd66c9a234ae02': 5.306, 'cbbcbc64c94ab911': 5.373, 'a65a456226d99617': 5.474, '9b305484155538ea': 5.691, '2d1e69e13898cded': 5.757, '76ae9c3d8ba52a1e': 5.88, '613f3f0987b63926': 5.937, 'c5c2d23adbbecc97': 6.053, 'ce64778d039fed0f': 6.279, '765fa209fd263054': 6.301, '435f38f3084d9e1e': 6.322, 'd941f3fd9b6c134e': 6.539, '7d8b37dbb8df1a4c': 6.718, '3f71c8ebe7618a46': 6.761, '656b5d577c2df30b': 6.877, 'e7d07b1d50efc23f': 6.893, '0a0732882784ca41': 6.898, '8864bd145f13b520': 6.944, '0cc272a5bc40647d': 7.04, 'c371918f6813863f': 7.078, 'a46fdef966f155a6': 7.202, '47a9a64cec09fd8a': 7.338, '2924dec2a3a29ff7': 7.461, 'c411ed18a2986624': 7.5, 'a22e8d823cd08d4d': 7.746, 'ebfd65ac1e8ae3b3': 7.754, 'cd6b120c89aeb032': 8.265, '2438e0dc74ca0da9': 8.528, 'd41a7b440d67b95c': 8.612, 'b45d006e147d2ec6': 8.702, 'c79f3a8bbb7ba4ee': 8.903, '5db1bf97d296fb7c': 8.936, '90ac6953b9ac02b9': 9.037, 'cb9bb18f1b5e255d': 9.17, '1d70e543323f9293': 9.174, 'ebb42c5ac374ea48': 9.472, 'a147cddfbd40fe2b': 9.599, 'c173026e187f0f52': 9.645, 'c2164275f6df47a1': 9.679, '53f44c540ebc8a61': 9.705, '04d31d1012d9c82c': 9.766, '67379540b569f702': 10.224, '84be13625f262d01': 10.335, '99f965d6728e96e1': 10.4, 'ee4ce65b8a1ed6e9': 10.471, '9ee896bb20202b84': 10.493, 'c512cf565d25e11d': 10.626, 'f7d9c51dd6a33aea': 10.657, 'e018f36a507012ec': 10.942, '671ba812d23461b0': 11.231, '6a624337306643cd': 11.501, '333289ce63f49e55': 11.746, '8e47b027ae97dd85': 12.188, 'bb9708e2a86bb28d': 12.19, '84652b2ab11c1077': 12.441, '81096303ad1d5136': 12.848, '60c95bdf8d6ac316': 13.606, 'c9112df988602506': 14.028, 'b2d5dd14d3252fa2': 14.583, '9df66cafa66e207e': 14.695, 'ddf177e019c3ea25': 14.784, '730a16f70c51f5e7': 15.069, '47efdc5033d2ec3c': 15.08, '34159ab10beaed5b': 15.164, 'dbbe848f94bebc52': 15.671, '2f67dc1961102e7a': 15.839, '324dd7dffb257e42': 16.237, '0897f1cd7a2c1a6f': 16.468, '2f418b8bd2683d77': 17.209, '2f0127d2e819e652': 17.685, '7f4490d70eb87eb2': 18.359, 'cbf54aabd697b916': 18.367, '6198b32a1ec6c93e': 18.597, 'b03393eec2b22e74': 18.692, '09ce3839956fd591': 19.269, '71d3e4849459123a': 20.048, '44369c4a6a3bf09f': 21.228, '263b2387e0a7d4c3': 21.44, 'a5660009252efdd7': 22.014, '0843b4b67f7e504f': 25.102, '4e96aa0cc4fab80c': 25.749, '0cc63eaccb63696c': 27.518, '10238525f06e897b': 27.991, '604980ea976ccf88': 28.784, '8302163cb6280531': 30.329, 'bd9a5774ac82b921': 31.272, 'b569536ba691e4d6': 35.406, '4ef627fefc564f60': 39.337, '79223df5df79d83f': 42.761, 'a12f2a323ebdff76': 50.453, '3af30ee882ddff05': 51.309}

# -- FIGURE generation for Figure 11 --
def figure_11_pre():
    plot_dir_ = f"{plot_dir}/steerability-examples"
    os.makedirs(plot_dir_, exist_ok=True)

    target_images = list(steerability_scores.keys())

    n_select = 4
    MIN_USER_SCORE_ = 100

    for target_image in target_images:
        MIN_USER_SCORE = MIN_USER_SCORE_
        
        df = user_df[user_df.target==target_image]
        if len(df) < n_select:
            continue
        target_meta = dataset[int(df.iloc[0].map_to_dataset)]
        target_prompt = target_meta['target_positive_prompt']

        best_score = list(df.groupby('user').apply(lambda x: x.improved_score.sum()).to_dict().items())
        try:
            best_score = sorted(best_score, key=lambda x: -x[1])
        except:
            continue
        best_score = [(k,v) for k,v in best_score]

        user_list = [k for k,_ in best_score]
        imgs = []
        all_imgs = []
        all_imgs_txt = []


        selected_i = set()
        while len(imgs) < n_select:
            MIN_USER_SCORE -= 10
            if MIN_USER_SCORE < 0:
                break
            for i in range(len(user_list)):
                df_i = df[df.user==user_list[i]].sort_values(['trajectory_index'])
                best_img = np.argmax(df_i.score)
                first_img = 0
                user_score = np.array(df_i.score)[best_img]
                scores = np.array(df_i.score)
                user_prompt = np.array(df_i.pos_prompt)[best_img]
                map_to_dataset = np.array(df_i.map_to_dataset)[best_img]
                img_meta = dataset[int(map_to_dataset)]
                embedding = np.array(img_meta['generated_image_embedding']['value'])

                first_user_score = np.array(df_i.score)[first_img]
                first_user_prompt = np.array(df_i.pos_prompt)[first_img]
                first_map_to_dataset = np.array(df_i.map_to_dataset)[first_img]
                first_img_meta = dataset[int(first_map_to_dataset)]
                first_embedding = np.array(first_img_meta['generated_image_embedding']['value'])

                text_embedding = text_embeddings[img_meta['generated_positive_prompt']]['CLIP']
                first_text_embedding = text_embeddings[first_img_meta['generated_positive_prompt']]['CLIP']

                if user_score >= MIN_USER_SCORE:
                    if i not in selected_i:
                        imgs.append({
                            'img':img_meta['generated_image'],
                            'prompt':user_prompt,
                            'score':user_score,
                            'embedding':embedding,
                            'text_embedding':text_embedding,

                            'first_img':first_img_meta['generated_image'],
                            'first_user_prompt': first_user_prompt,
                            'first_user_score': first_user_score,
                            'first_embedding':first_embedding,
                            'first_text_embedding':first_text_embedding
                        })
                        selected_i.add(i)
                for ei,map_to_dataset in enumerate(list(df_i.map_to_dataset)):
                    if ((ei == first_img) or (ei == best_img)) and (user_score >= MIN_USER_SCORE):
                        continue
                    if scores[ei] < 50:
                        continue
                    meta = dataset[int(map_to_dataset)]
                    if meta and 'generated_image_embedding' in meta:
                        embedding = np.array(img_meta['generated_image_embedding']['value'])
                        text_embedding = text_embeddings[img_meta['generated_positive_prompt']]['CLIP']
                        all_imgs.append(embedding)
                        all_imgs_txt.append(text_embedding)
        
        def find_furthest_vectors(vectors, k):
            n = len(vectors)
            if k >= n:
                return vectors, np.arange(len(vectors))
            vectors = np.array(vectors)

            # Calculate cosine similarity matrix
            similarity_matrix = np.dot(vectors, vectors.T) / (np.linalg.norm(vectors, axis=1) * np.linalg.norm(vectors, axis=1)[:, np.newaxis])

            max_distance = 1.0
            max_subset = None

            # Generate all combinations of k vectors
            rng = np.random.RandomState(0)
            max_n = 100
            vector_combinations = list(combinations(range(n), k))
            if max_n < len(vector_combinations):
                idxs = rng.choice(len(vector_combinations), size=max_n, replace=False)
                vector_combinations = [vector_combinations[idx] for idx in idxs]

            for combination in vector_combinations:
                # Calculate the pairwise cosine similarity for the current combination
                combination_similarity = similarity_matrix[np.ix_(combination, combination)]
                combination_similarity -= np.eye(len(combination_similarity))

                # Calculate the minimum similarity within the combination
                max_similarity = np.max(combination_similarity)

                if max_similarity < max_distance:
                    max_distance = max_similarity
                    max_subset = combination

            return [vectors[i] for i in max_subset], max_subset
        

        vectors_txt_ = [img['text_embedding'] for img in imgs]
        _, idxs = find_furthest_vectors(vectors_txt_, n_select)
        all_imgs.extend([img['embedding'] for idx,img in enumerate(imgs) if idx not in idxs])
        all_imgs.extend([img['first_embedding'] for idx,img in enumerate(imgs) if idx not in idxs])

        all_imgs_txt.extend([img['text_embedding'] for idx,img in enumerate(imgs) if idx not in idxs])
        all_imgs_txt.extend([img['first_text_embedding'] for idx,img in enumerate(imgs) if idx not in idxs])
        imgs = [img for idx,img in enumerate(imgs) if idx in idxs]
        imgs.append({'img':target_meta['target_image']})

        # Show image progression and target image
        fig = plt.figure(figsize=(24, 18))
        grid = gridspec.GridSpec(1, 2, width_ratios=[4.15, 1], hspace=0, wspace=0.25, top=1, bottom=0.3, left=0, right=1)
        gs1 = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=grid[0], wspace=0.05)
        gs2 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=grid[1], wspace=0)

        txts = []
        for im_i,img in enumerate(imgs):
            if (im_i < 4) and (im_i < len(imgs)-1):
                ax = plt.subplot(gs1[im_i])
            else:
                ax = plt.subplot(gs2[0])
            ax.imshow(img['img'])
            ax.axis('off')
            if (im_i < 4) and (im_i < len(imgs)-1):
                if 'prompt' not in img:
                    continue
                p = img['prompt']
                if "real,real,real," in p:
                    p1,p2 = p.split("real,real,real,")
                    p = p1 + "\nreal,real,real," + p2
            else:
                p = target_prompt
            
            if im_i < n_select:
                txt = ax.text(0.5, -0.025, f"$\\mathbf{{Score: {img['score']}}}$",
                            ha='center', va='top', transform=ax.transAxes, 
                            fontfamily='monospace', fontsize=14, wrap=True)
                txt2 = ax.text(0.5, -0.10, p,
                                ha='center', va='top', transform=ax.transAxes, 
                                fontfamily='monospace', fontsize=14, wrap=True)
                txt2._get_wrap_line_width = lambda : 275
                txts.append((txt, txt2))
            else:
                txt = ax.text(0.5, -0.025, f"$\\mathbf{{Steerability: {steerability_scores[target_image]:0.2f}}}$",
                            ha='center', va='top', transform=ax.transAxes, 
                            fontfamily='monospace', fontsize=14, wrap=True)
                txt2 = ax.text(0.5, -0.10, p,
                                ha='center', va='top', transform=ax.transAxes, 
                                fontfamily='monospace', fontsize=14, wrap=True)
                txt2._get_wrap_line_width = lambda : 275
                txts.append((txt, txt2))
        if len(txts) < 5:
            continue
            

        # Save
        filename = f"user_image--target-{target_image}"
        save_fig(filename=filename,
                    path=plot_dir_, 
                    exts=['jpg'],
                    fig=fig,
                    tight=False)
        
        filepath = f"{plot_dir_}/{filename}.jpg"
        print(filepath)
        save_with_cropped_whitespace(filepath)
    
def concatenate_images_vertically(image_list):
    # Calculate the maximum width and total height
    max_width = max(image.width for image in image_list)
    total_height = sum(image.height for image in image_list)

    # Create a new blank image with the calculated size
    new_image = Image.new('RGB', (max_width, total_height))

    # Paste each image vertically onto the new image
    y_offset = 0
    for image in image_list:
        new_image.paste(image, (0, y_offset))
        y_offset += image.height

    return new_image
def add_padding(image, padding):
    # Calculate the new size with padding
    new_width = image.width + 2 * padding
    new_height = image.height + 2 * padding

    # Add white padding to the image
    padded_image = ImageOps.expand(image, border=padding, fill='white')

    return padded_image

# -- FIGURE 11 --
def figure_11():
    selected_for_paper = ['b9b1c233f4e8a632', '08cdb3012619c11d', '99f965d6728e96e1', '671ba812d23461b0', '263b2387e0a7d4c3', '604980ea976ccf88']

    filepath = lambda x: f"{plot_dir}/steerability-examples/user_image--target-{x}.jpg"
    imgs = [load_image(filepath(sfpi), as_PIL=True) for sfpi in selected_for_paper]

    img = add_padding(concatenate_images_vertically(imgs), 20)

    filename = f"Figure 11"
    img.save(f"{plot_dir}/{filename}.jpg")
    save_with_cropped_whitespace(f"{plot_dir}/{filename}.jpg")
    IMG_HEIGHT = 414
    skip_rect_h0 = [13, 718, 1351, 2119, 2889, 3658]
    skip_rects=[
        [[sri, sri+IMG_HEIGHT], [1991, 2405]] for sri in skip_rect_h0
    ]
    save_with_colored_rectangle(f"{plot_dir}/{filename}.jpg", rect=[[0, None], [1981, 2415]], all=True, skip_rects=skip_rects, color=(0,255,0), alpha=0.2)


if __name__ == '__main__':
    if True:
        figure_49_pre()
        figure_4()
        figure_9_appendix()
    if False:
        figure_11_pre()
        figure_11()
    if False:
        figure_5()