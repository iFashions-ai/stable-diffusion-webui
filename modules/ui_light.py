import datetime
import mimetypes
import os
import random
import sys
import warnings
from functools import reduce

import gradio as gr
import gradio.utils
import numpy as np
from PIL import Image, PngImagePlugin  # noqa: F401

import modules.generation_parameters_copypaste as parameters_copypaste
import modules.gradio_hijack as grh
import modules.hypernetworks.ui as hypernetworks_ui
import modules.images
import modules.shared as shared
import modules.textual_inversion.textual_inversion as textual_inversion
import modules.textual_inversion.ui as textual_inversion_ui
from modules import gradio_extensons  # noqa: F401
from modules import (deepbooru, extra_networks, processing, progress,
                     prompt_parser, script_callbacks, scripts, sd_hijack,
                     sd_models, sd_samplers, shared_items, sysinfo, timer,
                     ui_checkpoint_merger, ui_common, ui_extensions,
                     ui_extra_networks, ui_light_html, ui_loadsave,
                     ui_postprocessing, ui_prompt_styles, ui_settings)
from modules.call_queue import (wrap_gradio_call, wrap_gradio_gpu_call,
                                wrap_queued_call)
from modules.generation_parameters_copypaste import image_from_url_text
from modules.paths import script_path
from modules.sd_hijack import model_hijack
from modules.shared import cmd_opts, opts
from modules.ui_common import create_refresh_button
from modules.ui_components import (FormGroup, FormHTML, FormRow,
                                   InputAccordion, ResizeHandleRow, ToolButton)
from modules.ui_gradio_extensions import reload_javascript

warnings.filterwarnings(
    "default" if opts.show_warnings else "ignore", category=UserWarning
)
warnings.filterwarnings(
    "default" if opts.show_gradio_deprecation_warnings else "ignore",
    category=gr.deprecation.GradioDeprecationWarning,
)

# this is a fix for Windows users. Without it, javascript files will be served with text/html content-type and the browser will not show any UI
mimetypes.init()
mimetypes.add_type("application/javascript", ".js")

# Likewise, add explicit content-type header for certain missing image types
mimetypes.add_type("image/webp", ".webp")


# Using constants for these since the variation selector isn't visible.
# Important that they exactly match script.js for tooltip to work.
random_symbol = '\U0001f3b2\ufe0f'  # 🎲️
reuse_symbol = '\u267b\ufe0f'  # ♻️
paste_symbol = '\u2199\ufe0f'  # ↙
refresh_symbol = '\U0001f504'  # 🔄
save_style_symbol = '\U0001f4be'  # 💾
apply_style_symbol = '\U0001f4cb'  # 📋
clear_prompt_symbol = '\U0001f5d1\ufe0f'  # 🗑️
extra_networks_symbol = '\U0001F3B4'  # 🎴
switch_values_symbol = '\U000021C5' # ⇅
restore_progress_symbol = '\U0001F300' # 🌀
detect_image_size_symbol = '\U0001F4D0'  # 📐


def calc_resolution_hires(enable, width, height, hr_scale, hr_resize_x, hr_resize_y):
    if not enable:
        return ""

    p = processing.StableDiffusionProcessingTxt2Img(width=width, height=height, enable_hr=True, hr_scale=hr_scale, hr_resize_x=hr_resize_x, hr_resize_y=hr_resize_y)
    p.calculate_target_resolution()

    return f"from <span class='resolution'>{p.width}x{p.height}</span> to <span class='resolution'>{p.hr_resize_x or p.hr_upscale_to_x}x{p.hr_resize_y or p.hr_upscale_to_y}</span>"


def create_sampler_and_steps_selection(choices, tabname):
    if opts.samplers_in_dropdown:
        with FormRow(elem_id=f"sampler_selection_{tabname}"):
            sampler_name = gr.Dropdown(label='Sampling method', elem_id=f"{tabname}_sampling", choices=choices, value=choices[0])
            steps = gr.Slider(minimum=1, maximum=150, step=1, elem_id=f"{tabname}_steps", label="Sampling steps", value=20)
    else:
        with FormGroup(elem_id=f"sampler_selection_{tabname}"):
            steps = gr.Slider(minimum=1, maximum=150, step=1, elem_id=f"{tabname}_steps", label="Sampling steps", value=20)
            sampler_name = gr.Radio(label='Sampling method', elem_id=f"{tabname}_sampling", choices=choices, value=choices[0])

    return steps, sampler_name


def ordered_ui_categories():
    user_order = {x.strip(): i * 2 + 1 for i, x in enumerate(shared.opts.ui_reorder_list)}

    for _, category in sorted(enumerate(shared_items.ui_reorder_categories()), key=lambda x: user_order.get(x[1], x[0] * 2 + 0)):
        yield category


def create_override_settings_dropdown(tabname, row):
    dropdown = gr.Dropdown([], label="Override settings", visible=False, elem_id=f"{tabname}_override_settings", multiselect=True)

    dropdown.change(
        fn=lambda x: gr.Dropdown.update(visible=bool(x)),
        inputs=[dropdown],
        outputs=[dropdown],
    )

    return dropdown
        

def create_ui():
    import modules.txt2img

    reload_javascript()

    demo = gr.Blocks(title="iFahsion AIGC Plateform", css=ui_light_html.css).queue()

    with demo as txt2img_interface:
        dummy_component = gr.Label(visible=False)
        with gr.Row():
            with gr.Column(scale=2):
                with gr.Row(elem_classes="type_row"):
                    with gr.Column(scale=17):
                        prompt = gr.Textbox(
                            show_label=False,
                            placeholder="Type prompt here.",
                            elem_id="txt2img_prompt",
                            container=False,
                            autofocus=True,
                            elem_classes=["type_row", "prompt"],
                            lines=4,
                        )

                        default_prompt = ""
                        if isinstance(default_prompt, str) and default_prompt != "":
                            txt2img_interface.load(lambda: default_prompt, outputs=prompt)

                    with gr.Column(scale=3, min_width=0):
                        generate_button = gr.Button(
                            label="Generate",
                            value="Generate",
                            elem_classes="type_row",
                            elem_id="txt2img_generate",
                            visible=True,
                            variant='primary'
                        )
                        skip_button = gr.Button(
                            label="Skip",
                            value="Skip",
                            elem_id="txt2img_skip",
                            elem_classes="type_row_half",
                            visible=False,
                        )
                        stop_button = gr.Button(
                            label="Stop",
                            value="Stop",
                            elem_classes="type_row_half",
                            elem_id="txt2img_interrupt",
                            visible=False,
                        )

                        def stop_clicked():
                            shared.state.interrupt()
                            return [gr.update(interactive=False)] * 2

                        stop_button.click(
                            stop_clicked,
                            outputs=[skip_button, stop_button],
                            queue=False,
                            _js="cancelGenerateForever",
                        )
                        skip_button.click(fn=lambda: shared.state.skip(), queue=False)
                
                (
                    txt2img_gallery,
                    generation_info,
                    html_info,
                    html_log,
                ) = ui_common.create_output_panel_light("txt2img", opts.outdir_txt2img_samples)

                default_advanced_checkbox = False
                with gr.Row(elem_classes="advanced_check_row"):
                    advanced_checkbox = gr.Checkbox(
                        label="Advanced",
                        value=default_advanced_checkbox,
                        container=False,
                        elem_classes="min_check",
                    )

            default_prompt_negative = ""
            with gr.Column(scale=1, visible=default_advanced_checkbox) as advanced_column:
                # with gr.Tab("Settings", id="txt2img_generation") as txt2img_generation_tab, ResizeHandleRow(equal_height=False):
                with gr.Column(variant='compact', elem_id="txt2img_settings"):
                    scripts.scripts_txt2img.prepare_ui()

                    negative_prompt = gr.Textbox(
                        label="Negative Prompt",
                        show_label=True,
                        placeholder="Type prompt here.",
                        info="Describing what you do not want to see.",
                        lines=2,
                        elem_id="txt2img_neg_prompt",
                        value=default_prompt_negative,
                    )

                    for category in ordered_ui_categories():
                        if category == "sampler":
                            steps, sampler_name = create_sampler_and_steps_selection(sd_samplers.visible_sampler_names(), "txt2img")

                        elif category == "dimensions":
                            with FormRow():
                                with gr.Column(elem_id="txt2img_column_size", scale=4):
                                    width = gr.Slider(minimum=64, maximum=2048, step=8, label="Width", value=512, elem_id="txt2img_width")
                                    height = gr.Slider(minimum=64, maximum=2048, step=8, label="Height", value=512, elem_id="txt2img_height")

                                with gr.Column(elem_id="txt2img_dimensions_row", scale=1, elem_classes="dimensions-tools"):
                                    res_switch_btn = ToolButton(value=switch_values_symbol, elem_id="txt2img_res_switch_btn", label="Switch dims")

                                if opts.dimensions_and_batch_together:
                                    with gr.Column(elem_id="txt2img_column_batch"):
                                        batch_count = gr.Slider(minimum=1, step=1, label='Batch count', value=1, elem_id="txt2img_batch_count")
                                        batch_size = gr.Slider(minimum=1, maximum=8, step=1, label='Batch size', value=1, elem_id="txt2img_batch_size")

                        elif category == "cfg":
                            with gr.Row():
                                cfg_scale = gr.Slider(minimum=1.0, maximum=30.0, step=0.5, label='CFG Scale', value=7.0, elem_id="txt2img_cfg_scale")

                        elif category == "checkboxes":
                            with FormRow(elem_classes="checkboxes-row", variant="compact"):
                                pass

                        elif category == "accordions":
                            with gr.Row(elem_id="txt2img_accordions", elem_classes="accordions"):
                                with InputAccordion(False, label="Hires. fix", elem_id="txt2img_hr") as enable_hr:
                                    with enable_hr.extra():
                                        hr_final_resolution = FormHTML(value="", elem_id="txtimg_hr_finalres", label="Upscaled resolution", interactive=False, min_width=0)

                                    with FormRow(elem_id="txt2img_hires_fix_row1", variant="compact"):
                                        hr_upscaler = gr.Dropdown(label="Upscaler", elem_id="txt2img_hr_upscaler", choices=[*shared.latent_upscale_modes, *[x.name for x in shared.sd_upscalers]], value=shared.latent_upscale_default_mode)
                                        hr_second_pass_steps = gr.Slider(minimum=0, maximum=150, step=1, label='Hires steps', value=0, elem_id="txt2img_hires_steps")
                                        denoising_strength = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label='Denoising strength', value=0.7, elem_id="txt2img_denoising_strength")

                                    with FormRow(elem_id="txt2img_hires_fix_row2", variant="compact"):
                                        hr_scale = gr.Slider(minimum=1.0, maximum=4.0, step=0.05, label="Upscale by", value=2.0, elem_id="txt2img_hr_scale")
                                        hr_resize_x = gr.Slider(minimum=0, maximum=2048, step=8, label="Resize width to", value=0, elem_id="txt2img_hr_resize_x")
                                        hr_resize_y = gr.Slider(minimum=0, maximum=2048, step=8, label="Resize height to", value=0, elem_id="txt2img_hr_resize_y")

                                    with FormRow(elem_id="txt2img_hires_fix_row3", variant="compact", visible=opts.hires_fix_show_sampler) as hr_sampler_container:

                                        hr_checkpoint_name = gr.Dropdown(label='Hires checkpoint', elem_id="hr_checkpoint", choices=["Use same checkpoint"] + modules.sd_models.checkpoint_tiles(use_short=True), value="Use same checkpoint")
                                        create_refresh_button(hr_checkpoint_name, modules.sd_models.list_models, lambda: {"choices": ["Use same checkpoint"] + modules.sd_models.checkpoint_tiles(use_short=True)}, "hr_checkpoint_refresh")

                                        hr_sampler_name = gr.Dropdown(label='Hires sampling method', elem_id="hr_sampler", choices=["Use same sampler"] + sd_samplers.visible_sampler_names(), value="Use same sampler")

                                    with FormRow(elem_id="txt2img_hires_fix_row4", variant="compact", visible=opts.hires_fix_show_prompts) as hr_prompts_container:
                                        with gr.Column(scale=80):
                                            with gr.Row():
                                                hr_prompt = gr.Textbox(label="Hires prompt", elem_id="hires_prompt", show_label=False, lines=3, placeholder="Prompt for hires fix pass.\nLeave empty to use the same prompt as in first pass.", elem_classes=["prompt"])
                                        with gr.Column(scale=80):
                                            with gr.Row():
                                                hr_negative_prompt = gr.Textbox(label="Hires negative prompt", elem_id="hires_neg_prompt", show_label=False, lines=3, placeholder="Negative prompt for hires fix pass.\nLeave empty to use the same negative prompt as in first pass.", elem_classes=["prompt"])

                                scripts.scripts_txt2img.setup_ui_for_section(category)

                        elif category == "batch":
                            if not opts.dimensions_and_batch_together:
                                with FormRow(elem_id="txt2img_column_batch"):
                                    batch_count = gr.Slider(minimum=1, step=1, label='Batch count', value=1, elem_id="txt2img_batch_count")
                                    batch_size = gr.Slider(minimum=1, maximum=8, step=1, label='Batch size', value=1, elem_id="txt2img_batch_size")

                        elif category == "override_settings":
                            with FormRow(elem_id="txt2img_override_settings_row") as row:
                                override_settings = create_override_settings_dropdown('txt2img', row)

                        elif category == "scripts":
                            with FormGroup(elem_id="txt2img_script_container"):
                                custom_inputs = scripts.scripts_txt2img.setup_ui()

                        if category not in {"accordions"}:
                            scripts.scripts_txt2img.setup_ui_for_section(category)

                hr_resolution_preview_inputs = [enable_hr, width, height, hr_scale, hr_resize_x, hr_resize_y]

                for component in hr_resolution_preview_inputs:
                    event = component.release if isinstance(component, gr.Slider) else component.change

                    event(
                        fn=calc_resolution_hires,
                        inputs=hr_resolution_preview_inputs,
                        outputs=[hr_final_resolution],
                        show_progress=False,
                    )
                    event(
                        None,
                        _js="onCalcResolutionHires",
                        inputs=hr_resolution_preview_inputs,
                        outputs=[],
                        show_progress=False,
                    )

            advanced_checkbox.change(
                lambda x: gr.update(visible=x),
                advanced_checkbox,
                advanced_column,
                queue=False,
            ).then(fn=lambda: None, _js="refresh_grid_delayed", queue=False)
            
            # Dummy bottons
            paste = ToolButton(value=paste_symbol, elem_id="paste", visible=False)
            dropdown = gr.Dropdown(visible=False, label="Styles", show_label=False, elem_id="txt2img_styles", choices=list(shared.prompt_styles.styles), value=[], multiselect=True, tooltip="Styles")
            
            txt2img_args = dict(
                fn=wrap_gradio_gpu_call(modules.txt2img.txt2img, extra_outputs=[None, '', '']),
                _js="submit",
                inputs=[
                    dummy_component,
                    prompt,
                    negative_prompt,
                    dropdown,
                    steps,
                    sampler_name,
                    batch_count,
                    batch_size,
                    cfg_scale,
                    height,
                    width,
                    enable_hr,
                    denoising_strength,
                    hr_scale,
                    hr_upscaler,
                    hr_second_pass_steps,
                    hr_resize_x,
                    hr_resize_y,
                    hr_checkpoint_name,
                    hr_sampler_name,
                    hr_prompt,
                    hr_negative_prompt,
                    override_settings,

                ] + custom_inputs,

                outputs=[
                    txt2img_gallery,
                    generation_info,
                    html_info,
                    html_log,
                ],
                show_progress=False,
            )

            generate_button.click(**txt2img_args)

            res_switch_btn.click(fn=None, _js="function(){switchWidthHeight('txt2img')}", inputs=None, outputs=None, show_progress=False)
            
            txt2img_paste_fields = [
                (prompt, "Prompt"),
                (negative_prompt, "Negative prompt"),
                (steps, "Steps"),
                (sampler_name, "Sampler"),
                (cfg_scale, "CFG scale"),
                (width, "Size-1"),
                (height, "Size-2"),
                (batch_size, "Batch size"),
                (dropdown, lambda d: d["Styles array"] if isinstance(d.get("Styles array"), list) else gr.update()),
                (denoising_strength, "Denoising strength"),
                (enable_hr, lambda d: "Denoising strength" in d and ("Hires upscale" in d or "Hires upscaler" in d or "Hires resize-1" in d)),
                (hr_scale, "Hires upscale"),
                (hr_upscaler, "Hires upscaler"),
                (hr_second_pass_steps, "Hires steps"),
                (hr_resize_x, "Hires resize-1"),
                (hr_resize_y, "Hires resize-2"),
                (hr_checkpoint_name, "Hires checkpoint"),
                (hr_sampler_name, "Hires sampler"),
                (hr_sampler_container, lambda d: gr.update(visible=True) if d.get("Hires sampler", "Use same sampler") != "Use same sampler" or d.get("Hires checkpoint", "Use same checkpoint") != "Use same checkpoint" else gr.update()),
                (hr_prompt, "Hires prompt"),
                (hr_negative_prompt, "Hires negative prompt"),
                (hr_prompts_container, lambda d: gr.update(visible=True) if d.get("Hires prompt", "") != "" or d.get("Hires negative prompt", "") != "" else gr.update()),
                *scripts.scripts_txt2img.infotext_fields
            ]
            parameters_copypaste.add_paste_fields("txt2img", None, txt2img_paste_fields, override_settings)
            parameters_copypaste.register_paste_params_button(parameters_copypaste.ParamBinding(
                paste_button=paste, tabname="txt2img", source_text_component=prompt, source_image_component=None,
            ))

            # txt2img_preview_params = [
            #     prompt,
            #     negative_prompt,
            #     steps,
            #     sampler_name,
            #     cfg_scale,
            #     scripts.scripts_txt2img.script('Seed').seed,
            #     width,
            #     height,
            # ]
            
    return demo