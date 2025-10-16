import matplotlib.pyplot as plt
import numpy as np
import pyrallis
import scipy.cluster.hierarchy as sch
import torch
import torchvision
import yaml
from config import Config
from manual_diffusion import ManualDiffusion
from pathlib import Path
from scipy.spatial.distance import pdist
from tqdm import tqdm
from utils import (
    load_prompts, downscale_height_and_width,
    set_seed, compute_mean_embeddings
)


@torch.no_grad()
@pyrallis.wrap()
def main(cfg: Config):
    # Seed
    set_seed(cfg.seed)

    Path(f"results/{cfg.exp_name}").mkdir(parents=True, exist_ok=True)

    # Save config
    with open(f"results/{cfg.exp_name}/config.yml", 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)

    # Initialize the manual diffusion model
    md = ManualDiffusion()
    rgb_height, rgb_width = 512, 512

    nn_dtype = md.weights_dtype

    prompts, image_embeddings = load_prompts(
        prompt_dataset=cfg.prompt_dataset,
        custom_prompts=cfg.custom_prompts,
        md=md
    )
    num_prompts = len(prompts)
    cfg.prompts = prompts
    if hasattr(md, 'prior_pipe'):
        print("Deleting prior pipe to save memory...")
        del md.prior_pipe
        torch.cuda.empty_cache()

    # Build the hierarchy
    print("Building hierarchy...")
    distance_matrix = pdist(image_embeddings[0].cpu().numpy(), metric="cosine")
    print("Computing linkage...")
    Z = sch.linkage(distance_matrix, method='ward', metric='cosine')

    if num_prompts <= cfg.max_batch_size:
        # Plot the dendrogram
        plt.figure(figsize=(10, 7))
        sch.dendrogram(Z, labels=cfg.prompts)
        # d_dict = sch.dendrogram(Z, labels=[prompt.replace("a ", "") for prompt in cfg.prompts])
        # d_dict = sch.dendrogram(Z, labels=cfg.prompts)
        plt.title('Embedding Dendrogram')
        plt.xlabel('Prompts')
        plt.xticks(rotation=90)
        plt.ylabel(r'$c^{score}$')
        plt.tight_layout()
        plt.savefig(f"results/{cfg.exp_name}/embedding_dendrogram.png")

    # Get the threshold distances / intervals
    phi = np.linspace(0, cfg.tau, cfg.inference_steps)[::-1]

    # Set+get total timesteps
    md.scheduler.set_timesteps(cfg.inference_steps, device=md.device)
    timesteps = md.scheduler.timesteps
    K = cfg.inference_steps

    # Initialize the latents
    num_channels_latents = md.unet.config.in_channels
    height, width = downscale_height_and_width(scale_factor=md.pipe.decoder_pipe.movq_scale_factor)
    x_k = md.pipe.decoder_pipe.prepare_latents(
            (1, num_channels_latents, height, width),
            image_embeddings[0].dtype,
            md.device,
            None,
            None,
            md.scheduler,
        )

    # Denoise the image
    total_step_ctr = 0
    prompt_tracker = np.array(cfg.prompts)
    C_prev = np.ones(num_prompts, dtype=int)  # Previous clusters
    for k in tqdm(range(K)):
        # Get the indices for each cluster at the current distance threshold
        phi_k = phi[k]

        # Get all clusters at c_score = phi_k (Lines 4-10 in algorithm 1)
        labels_k = sch.fcluster(Z, phi_k, criterion='distance')

        # Get unique clusters, inverse index, and first seen index
        C_k, first_index, inv, counts = np.unique(
            labels_k, return_inverse=True, return_index=True, return_counts=True)
        
        # Map first-seen parent for each cluster
        parent_map = dict(zip(C_k, C_prev[first_index]))

        # Update C_{k-1} aka C_prev
        C_prev = C_k[inv] # or labels_k

        # Get parent indices for each cluster
        C_parent = np.array([parent_map[c] for c in C_k]) - 1

        # # Duplicate the latents for each cluster
        x_k = x_k[C_parent]

        # Compute the mean embedding for each cluster
        inv_t = torch.from_numpy(inv).to(image_embeddings.device, dtype=torch.long)
        counts_t = torch.from_numpy(counts).to(image_embeddings.device, dtype=nn_dtype)
        y_hat = compute_mean_embeddings(
            embeddings=image_embeddings,
            inv=inv_t,
            counts=counts_t
        )

        # Get prompt indices for each cluster
        all_prompt_indices = []
        for c_idx in C_k:
            prompt_indices = np.where(inv == c_idx-1)[0]
            all_prompt_indices.append(prompt_indices)

        # Set the current conditional embeddings
        D = image_embeddings.size(2)
        current_image_embeddings = y_hat.reshape(-1, D)
        added_cond_kwargs = {"image_embeds": current_image_embeddings}

        # Keep track of prompt clusters origianl order
        prompt_clusters = []
        prompt_cluster_flat = []
        for group in all_prompt_indices:
            prompt_clusters.append(prompt_tracker[group])
            prompt_cluster_flat.append(", ".join(prompt_tracker[group]))

        if x_k.size(0) > cfg.max_batch_size:
            for batch in range(0, x_k.size(0), cfg.max_batch_size):
                pos_embeds = added_cond_kwargs["image_embeds"][batch:min(batch+cfg.max_batch_size, x_k.size(0))]
                neg_embeds = added_cond_kwargs["image_embeds"][x_k.size(0)+batch:x_k.size(0)+batch+cfg.max_batch_size]
                image_embeds = torch.cat([pos_embeds, neg_embeds])
                batch_added_cond_kwargs = {"image_embeds": image_embeds}

                # Denoise x_k for the current batch
                x_k[batch:batch+cfg.max_batch_size] = md.denoise(
                    x_k[batch:batch+cfg.max_batch_size],
                    text_embeddings=None,
                    t=timesteps[k],
                    added_cond_kwargs=batch_added_cond_kwargs,
                )
        else:
            x_k = md.denoise(
                x_k,
                text_embeddings=None,
                t=timesteps[k],
                added_cond_kwargs=added_cond_kwargs,
            )
        
        total_step_ctr += x_k.size(0)
        
    if x_k.size(0) > cfg.max_batch_size:
        denoised_images = torch.zeros(x_k.size(0), 3, rgb_height, rgb_width).to(md.device)
        for batch in range(0, x_k.size(0), cfg.max_batch_size):
            denoised_images[batch:batch+cfg.max_batch_size] = md.decode_latents(x_k[batch:batch+cfg.max_batch_size])
    else:
        denoised_images = md.decode_latents(x_k)

    # Reorder the denoised images based on the original prompt order
    original_indices = [prompt_cluster_flat.index(item) for item in cfg.prompts]
    prompt_cluster_flat = np.array(prompt_cluster_flat)

    # Save images in original order
    if denoised_images.size(0) > cfg.max_batch_size:
        batch_original_indices = original_indices[:cfg.max_batch_size]
        torchvision.utils.save_image(denoised_images[batch_original_indices],
            f"results/{cfg.exp_name}/denoised_images.png")
    else:
        torchvision.utils.save_image(denoised_images[original_indices],
            f"results/{cfg.exp_name}/denoised_images.png")
    torch.save(denoised_images[original_indices], f"results/{cfg.exp_name}/denoised_images.pt")

    stardard_total_steps = cfg.inference_steps * len(cfg.prompts)
    step_savings = f"Total steps: {total_step_ctr}, Standard total steps: {stardard_total_steps}"
    print(step_savings)
    with open(f"results/{cfg.exp_name}/savings.txt", "w") as f:
        f.write(step_savings)

if __name__ == "__main__":
    main()