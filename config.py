from dataclasses import dataclass
from typing import Optional, List
from dataclasses import field

@dataclass
class Config:
    exp_name: str = "custom_prompts"
    prompt_dataset: Optional[str] = None
    custom_prompts: Optional[List[str]] = None
    model_name: str = "kandinsky-community/kandinsky-2-2-decoder"
    prior_model_name: Optional[str] = "kandinsky-community/kandinsky-2-2-prior"
    guidance_scale: float = 10.0
    prior_guidance_scale: float = 4.0
    scheduler: str = "DDPM"
    inference_steps: int = 40
    tau: float = 1.0
    max_batch_size: int = 32
    seed: int = 42
