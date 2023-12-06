import os
import torch
from safetensors.torch import save_file
from pathlib import Path

import modules.shared as shared
from modules.modelloader import load_file_from_url


class InpaintHead(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.head = torch.nn.Parameter(torch.empty(size=(320, 5, 3, 3), device='cpu'))

    def __call__(self, x):
        x = torch.nn.functional.pad(x, (1, 1, 1, 1), "replicate")
        return torch.nn.functional.conv2d(input=x, weight=self.head)


inpaint_head_model = None


def patch_fooocus_inpaint(inpaint_latent, inpaint_latent_mask, model, inpaint_head_model_path):
    global inpaint_head_model

    if inpaint_head_model is None:
        inpaint_head_model = InpaintHead()
        sd = torch.load(inpaint_head_model_path, map_location='cpu')
        inpaint_head_model.load_state_dict(sd)

    scale_factor = 1  # 0.13025
    feed = torch.cat([
        inpaint_latent_mask,
        inpaint_latent * scale_factor
    ], dim=1)

    inpaint_head_model.to(device=feed.device, dtype=feed.dtype)
    inpaint_head_feature = inpaint_head_model(feed)

    def input_block_patch(h, index):
        if index == 0:
            h = h + inpaint_head_feature.to(h)
        return h

    m = model.clone()
    m.add_patch(input_block_patch, "input_block", "fooocus_inpaint_head")
    return m


def convert_patch_to_lora(patch_path):
    lora_file = Path(shared.models_path) / "Lora" / Path(patch_path).with_suffix(".safetensors").name
    if lora_file.exists():
        return str(lora_file)

    state_dict = torch.load(patch_path, map_location="cpu")

    new_state_dict = {}
    for k, v in state_dict.items():
        ks = k.split(".")
        module_name = "_".join(ks[:-1])
        param_name = ks[-1]
        for subname, value in zip(["w", "w_min", "w_max"], v):
            new_state_dict[f"{module_name}.{subname}.{param_name}"] = value

    save_file(new_state_dict, lora_file)
    return str(lora_file)


def download_fooocus_inpaint_models(version: str):
    assert version in ['v1', 'v2.5', 'v2.6']

    path_inpaint = os.path.join(shared.models_path, 'inpaint')
    load_file_from_url(
        url='https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/fooocus_inpaint_head.pth',
        model_dir=path_inpaint,
        file_name='fooocus_inpaint_head.pth'
    )
    head_file = os.path.join(path_inpaint, 'fooocus_inpaint_head.pth')

    if version == 'v1':
        load_file_from_url(
            url='https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/inpaint.fooocus.patch',
            model_dir=path_inpaint,
            file_name='inpaint.fooocus.patch'
        )
        patch_file = os.path.join(path_inpaint, 'inpaint.fooocus.patch')
    elif version == 'v2.5':
        load_file_from_url(
            url='https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/inpaint_v25.fooocus.patch',
            model_dir=path_inpaint,
            file_name='inpaint_v25.fooocus.patch'
        )
        patch_file = os.path.join(path_inpaint, 'inpaint_v25.fooocus.patch')
    elif version == 'v2.6':
        load_file_from_url(
            url='https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/inpaint_v26.fooocus.patch',
            model_dir=path_inpaint,
            file_name='inpaint_v26.fooocus.patch'
        )
        patch_file = os.path.join(path_inpaint, 'inpaint_v26.fooocus.patch')
    else:
        raise ValueError(f"Unknown fooocus_inpaint version: {version}")

    patch_file = convert_patch_to_lora(patch_file)
    return head_file, patch_file
