import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image, ImageChops
from transformers import CLIPModel, CLIPProcessor

from pipeline_rf_inversion_sde import RFInversionFluxPipelineSDE
import lpips
import timm  # NEW: DINO features

# ------------------------------------------------------------
# 1.  Metric helpers (CLIP-L/14@336, LPIPS, DINO-S/16)
# ------------------------------------------------------------
class MetricComputer:
    def __init__(self, device: torch.device):
        self.device = device

        # UPDATED: CLIP ViT-L/14@336
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14-336").to(device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14-336")

        # LPIPS on `device` (unchanged behavior)
        self.lpips_fn = lpips.LPIPS(net="vgg").to(device)
        self.lpips_transform = T.Compose(
            [T.Resize((1024, 1024)), T.ToTensor(), T.Normalize((0.5,), (0.5,))]
        )

        # NEW: DINO-S/16 backbone on `device`
        self.dino = timm.create_model("vit_small_patch16_224.dino", pretrained=True, num_classes=0).to(device).eval()
        self.dino_transform = T.Compose([
            T.Resize(256), T.CenterCrop(224), T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    @torch.inference_mode()
    def clip_similarity(self, image: Image.Image, text: str) -> float:
        inp = self.clip_processor(text=[text], images=image, return_tensors="pt", padding=True
                                  ).to(self.clip_model.device)
        return self.clip_model(**inp).logits_per_image.item()

    @torch.inference_mode()
    def lpips_distance(self, img1: Image.Image, img2: Image.Image) -> float:
        t1 = self.lpips_transform(img1).unsqueeze(0).to(self.device)
        t2 = self.lpips_transform(img2).unsqueeze(0).to(self.device)
        return self.lpips_fn(t1, t2).item()

    @torch.inference_mode()
    def dino_distance(self, img1: Image.Image, img2: Image.Image) -> float:
        x1 = self.dino_transform(img1).unsqueeze(0).to(self.device)
        x2 = self.dino_transform(img2).unsqueeze(0).to(self.device)
        f1 = F.normalize(self.dino(x1), dim=1)
        f2 = F.normalize(self.dino(x2), dim=1)
        cos_sim = F.cosine_similarity(f1, f2).item()
        return 1.0 - float(cos_sim)

    @staticmethod
    def pixel_distances(img1: Image.Image, img2: Image.Image) -> Tuple[float, float, float, float]:
        a1, a2 = np.asarray(img1, np.float32), np.asarray(img2, np.float32)
        diff   = a1 - a2
        l1_map = np.mean(np.abs(diff), axis=-1)        # (H, W)
        l2_map = np.mean(diff ** 2,  axis=-1)          # (H, W)
        return l1_map.mean(), np.median(l1_map), l2_map.mean(), np.median(l2_map)


# ------------------------------------------------------------
# 2.  RF‑Inversion FluxEditor (core kept untouched)
# ------------------------------------------------------------
class FluxEditor:
    def __init__(self, device: str = "cuda"):
        self.pipe = RFInversionFluxPipelineSDE.from_pretrained(
            "black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16
        )
        self.device = torch.device(device)
        # small CLIP + LPIPS only for internal prints (not needed for batch metrics)
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.lpips = lpips.LPIPS(net="vgg").to(self.device)
        self.lpips_transform = T.Compose(
            [T.Resize((1024, 1024)), T.ToTensor(), T.Normalize((0.5,), (0.5,))]
        )

    # ---- original helper (unchanged) ----
    def print_clip_score(self, image, prompt):
        inputs = self.clip_processor(text=[prompt], images=image, return_tensors="pt", padding=True)
        print("CLIP score:", self.clip_model(**inputs).logits_per_image.item())

    # ---- entire edit() copied verbatim from the demo ----
    @torch.inference_mode()
    def edit(
        self,
        init_image,
        gamma,
        target_prompt,
        target_guidance,
        num_steps,
        starting_index,
        stop_index,
        eta,
        target_prompt_2,
        seed,
    ):
        self.pipe = self.pipe.to(self.device)
        torch.cuda.empty_cache()
        if not seed:
            seed = torch.Generator(device="cpu").seed()
        print("Random seed:", seed)
        torch.manual_seed(int(seed))

        inverted_latents, image_latents, latent_image_ids = self.pipe.invert(
            image=init_image,
            num_inversion_steps=num_steps,
            gamma=gamma,
        )

        target_prompt_2 = target_prompt_2 if target_prompt_2 else None

        edited_image = self.pipe(
            prompt=target_prompt,
            prompt_2=target_prompt_2,
            inverted_latents=inverted_latents,
            image_latents=image_latents,
            latent_image_ids=latent_image_ids,
            start_timestep=starting_index / float(num_steps),
            stop_timestep=stop_index / float(num_steps),
            num_inference_steps=num_steps,
            eta=eta,
            enable_sde=True,
            guidance_scale=target_guidance,
        ).images[0]
        init_resized = init_image.resize(edited_image.size, Image.Resampling.LANCZOS)
        diff = ImageChops.difference(init_resized.convert("RGB"), edited_image.convert("RGB"))
        self.pipe = self.pipe.to("cpu")

        return edited_image, diff


# ------------------------------------------------------------
# 3.  Per‑example wrapper
# ------------------------------------------------------------
def run_edit(
    editor: FluxEditor,
    metric: MetricComputer,
    image_path: Path,
    target_prompt: str,
    *,
    save_dir: Path,
    idx: int,
    gamma: float,
    num_steps: int,
    target_guidance: float,
    starting_index: int,
    stop_index: int,
    eta: float,
    seed: int,
    target_prompt_2: str | None,
) -> Dict[str, float]:
    init_pil = Image.open(image_path).convert("RGB")

    edited, diff = editor.edit(
        init_pil,
        gamma,
        target_prompt,
        target_guidance,
        num_steps,
        starting_index,
        stop_index,
        eta,
        target_prompt_2,
        seed,
    )

    # ---- save images ----
    stem, ext = image_path.stem, image_path.suffix or ".png"
    save_dir.mkdir(parents=True, exist_ok=True)
    edit_path = save_dir / f"{idx:04d}_{stem}_edited{ext}"
    diff_path = save_dir / f"{idx:04d}_{stem}_diff{ext}"
    edited.save(edit_path)
    diff.save(diff_path)

    # ---- metrics ----
    clip_edit = metric.clip_similarity(edited, target_prompt)
    clip_src  = metric.clip_similarity(init_pil, target_prompt)
    lpips_val = metric.lpips_distance(init_pil, edited)
    dino_val  = metric.dino_distance(init_pil.resize(edited.size, Image.Resampling.LANCZOS), edited)
    l1_mean, l1_med, l2_mean, l2_med = metric.pixel_distances(init_pil.resize(edited.size), edited)

    return {
        "clip_target_edit": clip_edit,
        "clip_target_src":  clip_src,
        "lpips":            lpips_val,
        "dino":             dino_val,
        "l1_mean":          l1_mean,
        "l1_median":        l1_med,
        "l2_mean":          l2_mean,
        "l2_median":        l2_med,
        "edited_path":      str(edit_path),
        "diff_path":        str(diff_path),
        "seed":             seed,
    }


# ------------------------------------------------------------
# 4.  CLI & main loop
# ------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser("RF‑Inversion Experiment Runner")
    p.add_argument("--data", type=str, default="dataset.json")
    p.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")

    # demo defaults
    p.add_argument("--gamma",            type=float, default=0.5)
    p.add_argument("--num_steps",        type=int,   default=28)
    p.add_argument("--target_guidance",  type=float, default=3.5)
    p.add_argument("--starting_index",   type=int,   default=0)
    p.add_argument("--stop_index",       type=int,   default=7)
    p.add_argument("--eta",              type=float, default=0.9)
    p.add_argument("--target_prompt_2",  type=str,   default="")

    # misc
    p.add_argument("--save_dir", type=str, default="edited_rfinversion")
    p.add_argument("--csv_out",  type=str, default="results_rfinversion.csv")
    p.add_argument("--seed",     type=int, default=None, help="Global seed; blank for random per sample")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    editor = FluxEditor(device=args.device)
    metric = MetricComputer(device)

    with open(args.data) as fp:
        dataset: List[Dict] = json.load(fp)

    rows = []
    for idx, entry in enumerate(dataset):
        print(f"\n=== [{idx+1}/{len(dataset)}] {entry['image_path']} ===")
        seed = args.seed if args.seed is not None else int(torch.Generator(device="cpu").seed())

        stats = run_edit(
            editor,
            metric,
            Path(entry["image_path"]),
            entry["target_prompt"],              # only target prompt needed
            save_dir=Path(args.save_dir),
            idx=idx,
            gamma=args.gamma,
            num_steps=args.num_steps,
            target_guidance=args.target_guidance,
            starting_index=args.starting_index,
            stop_index=args.stop_index,
            eta=args.eta,
            seed=seed,
            target_prompt_2=args.target_prompt_2 or None,
        )
        for k, v in stats.items():
            print(f"{k:>20}: {v}")
        rows.append({**entry, **stats})

    out = Path(args.csv_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nFinished — detailed metrics stored in {out.resolve()}")


if __name__ == "__main__":
    main()
