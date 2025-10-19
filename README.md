# Reusing Computation in Text-to-Image Diffusion for Efficient Generation of Image Sets [ICCV 2025]
Authors: *[Dale Decatur](https://ddecatur.github.io/), [Thibault Groueix](https://imagine.enpc.fr/~groueixt/), [Yifan Wang](https://yifita.netlify.app/), [Rana Hanocka](https://people.cs.uchicago.edu/~ranahanocka/), [Vladimir G. Kim](http://vovakim.com/), [Matheus Gadelha](http://mgadelha.me/)*

\[[Paper](http://arxiv.org/abs/2508.21032)\] \[[Project Page](https://ddecatur.github.io/hierarchical-diffusion/)\]

![teaser](https://github.com/ddecatur/hierarchical-diffusion/raw/site/assets/tree_traversal.png)

### Abstract
Text-to-image diffusion models enable high-quality image generation but are computationally expensive. While prior work optimizes per-inference efficiency, we explore an orthogonal approach: reducing redundancy across correlated prompts. Our method leverages the coarse-to-fine nature of diffusion models, where early denoising steps capture shared structures among similar prompts. We propose a training-free approach that clusters prompts based on semantic similarity and shares computation in early diffusion steps. Experiments show that for models trained conditioned on image embeddings, our approach significantly reduces compute cost while improving image quality. By leveraging UnClip’s text-to-image prior, we enhance diffusion step allocation for greater efficiency. Our method seamlessly integrates with existing pipelines, scales with prompt sets, and reduces the environmental and financial burden of large-scale text-to-image generation.

## Install environment
Note: the packges involving cuda (i.e. PyTorch) must be installed on a GPU.

First create and activate the Conda enviroment
```
conda create -n hierarchical-diffusion python=3.12
conda activate hierarchical-diffusion
```

Install a PyTorch version compatible with your CUDA version. We use PyTorch 2.6 with CUDA 12.4.
```
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

Install the remaining required packages
```
pip install -r requirements.txt
```

## Run your own example!
To run the code on your own prompts, pass them in a list after the `--custom_prompts` command line argument.
```
python main.py --custom_prompts '["dog", "cat", "horse"]'
```
Feel free to also pass any of the fields in `config.py`. For example, to run an experiment called "my_experiment" with a tau value of 1.5, run the following code.
```
python main.py --custom_prompts '["dog", "cat", "horse"]' --exp_name my_experiment --tau 1.5
```
Any parameter in `config.py` can also be modified directly in that file without explicitly passing it as a command line argument. However, anything passed in the command line will overwrite the corresponding value in `config.py`.

After each run, the generated images are saved in `results/<exp_name>/` as `denoised_images.pt`. For runs with less than `max_batch_size` prompts, we also visualize the generated images as `denoised_images.png` and the corresponding dendrogram as `embedding_dendrogram.png`. If the number of prompts is greater than `max_batch_size`, we only visualize a `max_batch_size`-sized subset of the generated images and do not generate a dendrogram. For each run, we also store the config values used in `config.yml` and the savings relative to standard diffusion in `savings.txt`. See below for a visualization of the file structure.
```
...
├── results/
│   └── <exp_name>/
│       ├── config.yml
│       ├── denoised_images.png
│       ├── denoised_images.pt
│       ├── embedding_dendrogram.png
│       └── savings.txt
...
```

## Comparing with standard diffusion inference
To run standard diffusion, set Tau to $0$ either with the command line argument or in `config.py`.
```
python main.py --custom_prompts '["dog", "cat", "horse"]' --exp_name standard_diffusion --tau 0
```
Since setting $\tau = 0$ maps all diffusion steps to $c^{score} = 0$ (per eq $7$ from the paper shown below), no computation is shared and our approach becomes identical to standard diffusion inference.
$\begin{equation}
    \phi(k) = \tau \cdot \left(1 - \frac{k}{K}\right) \tag{7}
\end{equation}$

## Reproduce paper results
To run our method on the datasets we use in the paper (`animals`, `genai_bench`, `prompt_templates`, and `style_variations`), pass the dataset name in the `--prompt_dataset` command line argument or directly set the `prompt_dataset` field in `config.py`. For table $1$, we use the following commands.
```
python main.py --prompt_dataset genai_bench --exp_name genai_bench --tau 1.0
python main.py --prompt_dataset prompt_templates --exp_name prompt_templates --tau 1.0
python main.py --prompt_dataset style_variations --exp_name style_variations --tau 1.0
python main.py --prompt_dataset animals --exp_name animals --tau 1.0
```

## Acknowledgements
We thank Richard Zhang for insights on evaluation metrics and more generally, the members of Adobe Research and 3DL for their insightful feedback. This work was supported by Adobe Research and NSF grant 2140001.

## Citation
If you find this code helpful for your research, please cite our paper
[Reusing Computation in Text-to-Image Diffusion for Efficient Generation of Image Sets](http://arxiv.org/abs/2508.21032).
```
@inproceedings{decatur2025reusing,
  title = {Reusing Computation in Text-to-Image Diffusion for Efficient Generation of Image Sets},
  author = {Decatur, Dale and Groueix, Thibault and Yifan, Wang and Hanocka, Rana and Kim, Vladimir and Gadelha, Matheus},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year = {2025}
}
