import torch
import torch.nn.functional as F
import gradio as gr
from PIL import ImageChops, Image
import numpy as np

from pipeline_rf_inversion_sde import RFInversionFluxPipelineSDE

from transformers import CLIPProcessor, CLIPModel
import lpips
import torchvision.transforms as transforms

import timm

class FluxEditor:
    def __init__(self):
        self.pipe = RFInversionFluxPipelineSDE.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            torch_dtype=torch.bfloat16,
        )
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14-336") # it's lightweighted so it can live on CPU
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
        self.lpips = lpips.LPIPS(net='vgg').to('cuda')
        self.lpips_transform = transforms.Compose([
            transforms.Resize((1024, 1024)),  # resize for consistency (optional)
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # LPIPS expects inputs in [-1, 1]
        ])
        
        self.dino = timm.create_model('vit_small_patch16_224.dino', pretrained=True, num_classes=0).to('cuda')
        self.dino.eval()
        self.dino_transform = transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def dino_dist(self, img1, img2):
        img1 = self.dino_transform(img1).unsqueeze(0)
        img2 = self.dino_transform(img2).unsqueeze(0)
        with torch.no_grad():
            emb1 = F.normalize(self.dino(img1.to('cuda')), dim=1)
            emb2 = F.normalize(self.dino(img2.to('cuda')), dim=1)
        return F.cosine_similarity(emb1, emb2).item()

    def print_clip_score(self, image, prompt):
        clip_inputs = self.clip_processor(
            text=[prompt,],
            images=image,
            return_tensors="pt",
            padding=True,
        )
        clip_outputs = self.clip_model(**clip_inputs)
        print("CLIP score: ", clip_outputs.logits_per_image.detach().item())
    
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
        self.pipe = self.pipe.to("cuda")
        torch.cuda.empty_cache()
        if not seed:
            seed = torch.Generator(device="cpu").seed()
        print("Random seed: ", seed)
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
            start_timestep=starting_index/float(num_steps),
            stop_timestep=stop_index/float(num_steps),
            num_inference_steps=num_steps,
            eta=eta,    
            enable_sde=True,
            guidance_scale=target_guidance,
        ).images[0]
        init_resized = init_image.resize(
            edited_image.size,              # match W×H
            Image.Resampling.LANCZOS        # high-quality down/up-sampling
        )
        diff_img = ImageChops.difference(
            init_resized.convert("RGB"),   # make sure both are RGB
            edited_image.convert("RGB")
        )
        self.pipe = self.pipe.to("cpu")

        print("target prompt vs target image: ")
        self.print_clip_score(edited_image, target_prompt)
        print("target prompt vs source image: ")
        self.print_clip_score(init_resized, target_prompt)
        
        a1 = np.asarray(init_resized, dtype=np.float32)
        a2 = np.asarray(edited_image, dtype=np.float32)
        # per‑channel difference
        diff_np = a1 - a2
        # average across the 3 channels so l1/l2 match image (H, W)
        l1 = np.mean(np.abs(diff_np), axis=-1)      # shape (H, W)
        l2 = np.mean(diff_np ** 2,  axis=-1)        # shape (H, W)

        print("L1 Distance:", l1.mean(), "median:", np.median(l1))
        print("L2 Distance:", l2.mean(), "median:", np.median(l2))

        
        with torch.no_grad():
            dist = self.lpips(self.lpips_transform(init_resized).to("cuda"), self.lpips_transform(edited_image).to("cuda"))
        print("LPIPS distance: ", dist.item())
        
        with torch.no_grad():
            dist = self.dino_dist(init_resized, edited_image)
        print("DINO distance: ", 1.0-dist)

        torch.cuda.empty_cache()
        print("End Edit\n\n")
        return edited_image, diff_img



def create_demo(model_name: str, device: str = "cuda" if torch.cuda.is_available() else "cpu", offload: bool = False):
    editor = FluxEditor()

    with gr.Blocks() as demo:
        gr.Markdown(f"RF-Inversion Demo")
        
        with gr.Row():
            with gr.Column():
                target_prompt = gr.Textbox(label="Target Prompt. RF-inversion does not need source prompt.", value="")
                target_guidance = gr.Slider(0.0, 10.0, 1.0, step=0.05, label="Target Guidance")
                num_steps = gr.Slider(1, 100, 28, step=1, label="Number of steps")
                generate_btn = gr.Button("Generate")
                
                
                with gr.Accordion("Advanced Options", open=True):
                    starting_index = gr.Slider(0, 100, 0, step=1, label="starting index")
                    stop_index = gr.Slider(0, 100, 7, step=1, label="stop index")
                    gamma = gr.Slider(0.0, 10.0, 0.5, step=0.05, label="gamma Guidance")
                    eta = gr.Slider(0.0, 10.0, 0.9, step=0.05, label="eta Guidance")
                    seed = gr.Textbox(None, label="Seed")
                    target_prompt_2 = gr.Textbox(label="Target Prompt 2 for t5 encoder", value=None)
            
            with gr.Column():
                init_image = gr.Image(label="Input Image", visible=True, type='pil')
            
            with gr.Column():
                output_image = gr.Image(label="Generated Image", format='jpg')
            
            with gr.Column():
                diff_image   = gr.Image(label="Difference (|input - output|)", format='jpg')

        generate_btn.click(
            fn=editor.edit,
            inputs=[        
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
            ],
            outputs=[output_image, diff_image]
        )


    return demo


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Flux")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--offload", action="store_true", help="Offload model to CPU when not in use")
    parser.add_argument("--share", action="store_true", help="Create a public link to your demo")

    parser.add_argument("--port", type=int, default=49035)
    args = parser.parse_args()

    demo = create_demo("SDE coupling Demo with Flux", args.device, args.offload)
    demo.launch(server_name='0.0.0.0', share=args.share, server_port=args.port)
