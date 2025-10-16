import torch
from config import Config
from diffusers import (
    AutoPipelineForText2Image,
    KandinskyV22PriorPipeline,
    DDPMScheduler,
)

class ManualDiffusion:
    def __init__(self, cfg: Config = None) -> None:
        # Set device (cuda or cpu)
        self.device_name = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(self.device_name)

        # If no config passed, use default config
        if cfg is None:
            cfg = Config()
        self.cfg = cfg

        # Load model
        self.weights_dtype = torch.float16
        self.load_model()

    def load_model(self):
        # Load prior model
        self.prior_pipe = KandinskyV22PriorPipeline.from_pretrained(
            self.cfg.prior_model_name,
            torch_dtype=self.weights_dtype).to(self.device)
        # Load main model
        self.pipe = AutoPipelineForText2Image.from_pretrained(
            self.cfg.model_name,
            torch_dtype=self.weights_dtype).to(self.device)
        
        self.unet = self.pipe.unet.eval()
        for p in self.unet.parameters():
            p.requires_grad_(False)
        self.scheduler = self.pipe.scheduler

        if self.cfg.scheduler == "DDPM":
            self.scheduler = DDPMScheduler.from_config(self.pipe.scheduler.config)

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps

        self.alphas: torch.FloatTensor = self.scheduler.alphas_cumprod.to(
            self.device
        )
    
    def decode_latents(self, latents):
        input_dtype = latents.dtype
        latents = latents.to(self.weights_dtype)
        image = self.pipe.movq.decode(latents, force_not_quantize=True)["sample"]
        image = image * 0.5 + 0.5
        image = image.clamp(0, 1)
        return image.to(input_dtype)

    def denoise(
        self,
        z_t: torch.FloatTensor,
        text_embeddings: torch.FloatTensor,
        t: torch.FloatTensor,
        added_cond_kwargs = None,
    ):
        with torch.no_grad():
            latent_model_input = torch.cat([z_t] * 2, dim=0)

            noise_pred = self.unet(
                sample=latent_model_input,
                timestep=t,
                encoder_hidden_states=text_embeddings,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]
            
            # Perform classifier-free guidance
            noise_pred, variance_pred = noise_pred.split(z_t.shape[1], dim=1)
            noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
            variance_pred_text, _ = variance_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.cfg.guidance_scale * (noise_pred_text - noise_pred_uncond)
            noise_pred = torch.cat([noise_pred, variance_pred_text], dim=1)
            if not (
                hasattr(self.scheduler.config, "variance_type")
                and self.scheduler.config.variance_type in ["learned", "learned_range"]
            ):
                noise_pred, _ = noise_pred.split(z_t.shape[1], dim=1)

            # compute the previous noisy sample x_t -> x_t-1
            z_t = self.scheduler.step(
                noise_pred,
                t,
                z_t,
                generator=None,
            )[0]

        return z_t
