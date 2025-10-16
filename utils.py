import numpy as np
import os
import random
import torch
from pathlib import Path
from tqdm import tqdm

# seed everything
def set_seed(seed):
    """Seed for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def compute_mean_embeddings(embeddings, inv, counts):
    """
    Batched deterministic PyTorch code to compute
    mean embeddings for each cluster.
    """
    # Cast to float32 avoid fp errors
    input_dtype = embeddings.dtype
    embeddings = embeddings.to(torch.float32)
    counts = counts.to(torch.float32)
    # Sort the embeddings according to the cluster order
    cluster_order = torch.argsort(inv, stable=True)
    grouped_y = embeddings[:, cluster_order]
    csum_counts = torch.cumsum(counts.to(torch.long), dim=0)
    ends = csum_counts - 1
    starts = torch.cat([torch.zeros_like(csum_counts[:1]), csum_counts[:-1]], dim=0)

    csum = torch.cumsum(grouped_y, dim=1)
    prev = torch.cat([torch.zeros_like(csum[:, :1]), csum[:, :-1]], dim=1)
    sums = csum[:, ends] - prev[:, starts]

    y_hat = sums / counts[None, :, None]
    return y_hat.to(input_dtype)

def downscale_height_and_width(height=512, width=512, scale_factor=8):
    new_height = height // scale_factor**2
    if height % scale_factor**2 != 0:
        new_height += 1
    new_width = width // scale_factor**2
    if width % scale_factor**2 != 0:
        new_width += 1
    return new_height * scale_factor, new_width * scale_factor

def compute_prior(md, prompt):
    if md.cfg.prior_model_name == "kakaobrain/karlo-v1-alpha":
        _ = md.prior_pipe(prompt).to_tuple()[0]
        pos_emb = md.prior_pipe.image_embeddings
        neg_emg = torch.zeros_like(pos_emb)
    else:
        pos_emb, neg_emg = md.prior_pipe(prompt).to_tuple()
    return torch.cat([pos_emb, neg_emg]).to(md.weights_dtype)

def load_prompts(prompt_dataset=None, custom_prompts=None, md=None):
    assert (prompt_dataset is not None) or (custom_prompts is not None), \
        "Either prompt_dataset or custom_prompts must be provided."
    if custom_prompts is not None:
        prompts = custom_prompts
        prompt_dataset = "custom"
    else:
        assert prompt_dataset in [
            "animals", "style_variations",
            "prompt_templates", "genai_bench", "demo"], \
            f"Unknown prompt dataset: {prompt_dataset}"
        prompts = np.load(f"prompts/{prompt_dataset}.npy")
    if Path(f"prompts/{prompt_dataset}_embeddings.pt").exists() and prompt_dataset != "custom":
        print("Loading cached embeddings...")
        image_embeddings = torch.load(f"prompts/{prompt_dataset}_embeddings.pt")
    else:
        print("Computing embeddings from scratch...")
        assert md is not None, \
            "ManualDiffusion instance must be passed if embeddings are not cached."
        image_embeddings = []
        for prompt in tqdm(prompts):
            image_embeddings.append(compute_prior(md, prompt))
        del md.prior_pipe
        torch.cuda.empty_cache()
        image_embeddings = torch.stack(image_embeddings, dim=1)
        torch.save(image_embeddings, f"prompts/{prompt_dataset}_embeddings.pt")    
    return prompts, image_embeddings
