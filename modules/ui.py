import datetime
import mimetypes
import os
import sys
from functools import reduce
import warnings
import math

import gradio as gr
import gradio.utils
import numpy as np
from PIL import Image, PngImagePlugin  # noqa: F401
from modules.call_queue import wrap_gradio_gpu_call, wrap_queued_call, wrap_gradio_call

from modules import gradio_extensons  # noqa: F401
from modules import sd_hijack, sd_models, script_callbacks, ui_extensions, deepbooru, extra_networks, ui_common, ui_postprocessing, progress, ui_loadsave, shared_items, ui_settings, timer, sysinfo, ui_checkpoint_merger, ui_prompt_styles, scripts, sd_samplers, processing, ui_extra_networks
from modules.ui_components import FormRow, FormColumn, FormGroup, ToolButton, FormHTML, InputAccordion, ResizeHandleRow
from modules.paths import script_path
from modules.ui_common import create_refresh_button
from modules.ui_gradio_extensions import reload_javascript

from modules.shared import opts, cmd_opts

import modules.generation_parameters_copypaste as parameters_copypaste
import modules.hypernetworks.ui as hypernetworks_ui
import modules.textual_inversion.ui as textual_inversion_ui
import modules.textual_inversion.textual_inversion as textual_inversion
import modules.shared as shared
import modules.images
from modules import prompt_parser
from modules.sd_hijack import model_hijack
from modules.generation_parameters_copypaste import image_from_url_text

create_setting_component = ui_settings.create_setting_component

warnings.filterwarnings("default" if opts.show_warnings else "ignore", category=UserWarning)
warnings.filterwarnings("default" if opts.show_gradio_deprecation_warnings else "ignore", category=gr.deprecation.GradioDeprecationWarning)

# this is a fix for Windows users. Without it, javascript files will be served with text/html content-type and the browser will not show any UI
mimetypes.init()
mimetypes.add_type('application/javascript', '.js')

# Likewise, add explicit content-type header for certain missing image types
mimetypes.add_type('image/webp', '.webp')

if not cmd_opts.share and not cmd_opts.listen:
    # fix gradio phoning home
    gradio.utils.version_check = lambda: None
    gradio.utils.get_local_ip_address = lambda: '127.0.0.1'

if cmd_opts.ngrok is not None:
    import modules.ngrok as ngrok
    print('ngrok authtoken detected, trying to connect...')
    ngrok.connect(
        cmd_opts.ngrok,
        cmd_opts.port if cmd_opts.port is not None else 7860,
        cmd_opts.ngrok_options
        )


def gr_show(visible=True):
    return {"visible": visible, "__type__": "update"}


sample_img2img = "assets/stable-samples/img2img/sketch-mountains-input.jpg"
sample_img2img = sample_img2img if os.path.exists(sample_img2img) else None

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


plaintext_to_html = ui_common.plaintext_to_html

# Define aspect ratios
# sdxl_aspect_ratios = [
#     '704x1408', '704x1344', '768x1344', '768x1280', '832x1216', '832x1152',
#     '896x1152', '896x1088', '960x1088', '960x1024', '1024x1024', '1024x960',
#     '1088x960', '1088x896', '1152x896', '1152x832', '1216x832', '1280x768',
#     '1344x768', '1344x704', '1408x704', '1472x704', '1536x640', '1600x640',
#     '1664x576', '1728x576'
# ]
sdxl_aspect_ratios = [
    '704x1408', '768x1344', '768x1280',
    '896x1152', '960x1024', '1024x1024', '1024x960',
    '1152x896', '1280x768',
    '1344x768', '1408x704', '1536x640', '1600x640',
    '1664x576', '1728x576'
]
sd15_aspect_ratios = [
    '512x512', '640x640', '576x704'
]

def add_ratio(x):
    a, b = x.replace('x', ' ').split(' ')[:2]
    a, b = int(a), int(b)
    g = math.gcd(a, b)
    return f'{a}×{b} | {a // g}:{b // g}'


sdxl_aspect_ratios = list(map(add_ratio, sdxl_aspect_ratios))
sd15_aspect_ratios = list(map(add_ratio, sd15_aspect_ratios))


def send_gradio_gallery_to_image(x):
    if len(x) == 0:
        return None
    return image_from_url_text(x[0])


def calc_resolution_hires(enable, width, height, hr_scale, hr_resize_x, hr_resize_y):
    if not enable:
        return ""

    p = processing.StableDiffusionProcessingTxt2Img(width=width, height=height, enable_hr=True, hr_scale=hr_scale, hr_resize_x=hr_resize_x, hr_resize_y=hr_resize_y)
    p.calculate_target_resolution()

    return f"from <span class='resolution'>{p.width}x{p.height}</span> to <span class='resolution'>{p.hr_resize_x or p.hr_upscale_to_x}x{p.hr_resize_y or p.hr_upscale_to_y}</span>"


def resize_from_to_html(width, height, scale_by):
    target_width = int(width * scale_by)
    target_height = int(height * scale_by)

    if not target_width or not target_height:
        return "no image selected"

    return f"resize: from <span class='resolution'>{width}x{height}</span> to <span class='resolution'>{target_width}x{target_height}</span>"


def process_interrogate(interrogation_function, mode, ii_input_dir, ii_output_dir, *ii_singles):
    if mode in {0, 1, 3, 4}:
        return [interrogation_function(ii_singles[mode]), None]
    elif mode == 2:
        return [interrogation_function(ii_singles[mode]["image"]), None]
    elif mode == 5:
        assert not shared.cmd_opts.hide_ui_dir_config, "Launched with --hide-ui-dir-config, batch img2img disabled"
        images = shared.listfiles(ii_input_dir)
        print(f"Will process {len(images)} images.")
        if ii_output_dir != "":
            os.makedirs(ii_output_dir, exist_ok=True)
        else:
            ii_output_dir = ii_input_dir

        for image in images:
            img = Image.open(image)
            filename = os.path.basename(image)
            left, _ = os.path.splitext(filename)
            print(interrogation_function(img), file=open(os.path.join(ii_output_dir, f"{left}.txt"), 'a', encoding='utf-8'))

        return [gr.update(), None]


def interrogate(image):
    prompt = shared.interrogator.interrogate(image.convert("RGB"))
    return gr.update() if prompt is None else prompt


def interrogate_deepbooru(image):
    prompt = deepbooru.model.tag(image)
    return gr.update() if prompt is None else prompt


def connect_clear_prompt(button):
    """Given clear button, prompt, and token_counter objects, setup clear prompt button click event"""
    button.click(
        _js="clear_prompt",
        fn=None,
        inputs=[],
        outputs=[],
    )


def update_token_counter(text, steps, *, is_positive=True):
    try:
        text, _ = extra_networks.parse_prompt(text)

        if is_positive:
            _, prompt_flat_list, _ = prompt_parser.get_multicond_prompt_list([text])
        else:
            prompt_flat_list = [text]

        prompt_schedules = prompt_parser.get_learned_conditioning_prompt_schedules(prompt_flat_list, steps)

    except Exception:
        # a parsing error can happen here during typing, and we don't want to bother the user with
        # messages related to it in console
        prompt_schedules = [[[steps, text]]]

    flat_prompts = reduce(lambda list1, list2: list1+list2, prompt_schedules)
    prompts = [prompt_text for step, prompt_text in flat_prompts]
    token_count, max_length = max([model_hijack.get_prompt_lengths(prompt) for prompt in prompts], key=lambda args: args[0])
    return f"<span class='gr-box gr-text-input'>{token_count}/{max_length}</span>"


def update_negative_prompt_token_counter(text, steps):
    return update_token_counter(text, steps, is_positive=False)


class PromptColumn:
    def __init__(self, is_img2img: bool):
        id_part = "img2img" if is_img2img else "txt2img"
        self.id_part = id_part

        with gr.Column(elem_id=f"{id_part}_prompt_container"):
            # Input Row
            with gr.Row():
                with gr.Column(scale=17):
                    self.prompt = gr.Textbox(label="Prompt", elem_id=f"{id_part}_prompt", show_label=False, lines=4, placeholder="Type your prompt here", elem_classes=["prompt"])

                with gr.Column(scale=3, elem_id=f"{id_part}_actions_column"):
                    with gr.Row():
                        self.interrupt = gr.Button('Stop', elem_id=f"{id_part}_interrupt", elem_classes="generate-box-interrupt")
                        self.skip = gr.Button('Skip', elem_id=f"{id_part}_skip", elem_classes="generate-box-skip")
                        self.submit = gr.Button('Generate', elem_id=f"{id_part}_generate", variant='primary')

                        self.skip.click(
                            fn=lambda: shared.state.skip(),
                            inputs=[],
                            outputs=[],
                        )

                        self.interrupt.click(
                            fn=lambda: shared.state.interrupt(),
                            inputs=[],
                            outputs=[],
                        )


class AdvancedColumn:
    def __init__(self, is_img2img: bool, interface, gallery):
        id_part = "img2img" if is_img2img else "txt2img"
        self.id_part = id_part

        scripts_runner = scripts.scripts_img2img if is_img2img else scripts.scripts_txt2img

        with gr.Tabs(elem_id=f"{self.id_part}_extra_tabs"):
            default_prompt_negative = ""
            with gr.Tab("Configuration", id=f"{self.id_part}_generation", render=True) as configuration_tab:
                with gr.Column(variant='compact', elem_id=f"{self.id_part}_settings"):
                    with gr.Row():
                        self.negative_prompt = gr.Textbox(
                            label="Negative Prompt",
                            show_label=True,
                            placeholder="Type what you don't want to see here",
                            info="Describing what you do not want to see.",
                            lines=2,
                            elem_id=f"{self.id_part}_neg_prompt",
                            value=default_prompt_negative,
                        )
                        self.paste = ToolButton(value=paste_symbol, elem_id="paste", tooltip="Read generation parameters from prompt or last generation if prompt is empty into user interface.")

                    for category in ordered_ui_categories():
                        if category == "sampler":
                            self.steps, self.sampler_name = create_sampler_and_steps_selection(sd_samplers.visible_sampler_names(), tabname=id_part, sampler_interactive=True)

                        elif category == "dimensions":
                            aspect_ratios = sdxl_aspect_ratios if opts.data.get("sdxl_filter_enabled", True) else sd15_aspect_ratios
                            self.aspect_ratios_selection = gr.Radio(label='Aspect Ratios', choices=aspect_ratios,
                                                value=None, info='width × height',
                                                elem_classes='aspect_ratios')
                            with FormRow():
                                with gr.Column(elem_id=f"{self.id_part}_column_size", scale=4):
                                    self.width = gr.Slider(minimum=64, maximum=2048, step=8, label="Width", value=512, elem_id=f"{self.id_part}_width")
                                    self.height = gr.Slider(minimum=64, maximum=2048, step=8, label="Height", value=512, elem_id=f"{self.id_part}_height")

                                with gr.Column(elem_id=f"{self.id_part}_dimensions_row", scale=1, elem_classes="dimensions-tools"):
                                    self.res_switch_btn = ToolButton(value=switch_values_symbol, elem_id=f"{self.id_part}_res_switch_btn", label="Switch dims")
                                    if self.id_part == "img2img":
                                        self.detect_image_size_btn = ToolButton(value=detect_image_size_symbol, elem_id=f"{self.id_part}_detect_image_size_btn", tooltip="Auto detect size from img2img")
                                if opts.dimensions_and_batch_together:
                                    with gr.Column(elem_id=f"{self.id_part}_column_batch"):
                                        self.batch_count = gr.Slider(minimum=1, maximum=32, step=1, label='Batch count', value=1, elem_id=f"{self.id_part}_batch_count")
                                        self.batch_size = gr.Slider(visible=False, minimum=1, maximum=8, step=1, label='Batch size', value=1, elem_id=f"{self.id_part}_batch_size")

                            if self.id_part == "img2img":
                                self.resize_mode = gr.Radio(label="Resize mode", elem_id="resize_mode", choices=["Just resize", "Crop and resize", "Resize and fill", "Just resize (latent upscale)"], type="index", value="Resize and fill")

                            def update_aspect_ratio(value):
                                if not value:
                                    width, height = 1024, 1024
                                else:
                                    width, height = map(int, value.split()[0].split("×"))
                                return gr.Slider.update(value=width), gr.Slider.update(value=height)

                            self.aspect_ratios_selection.change(update_aspect_ratio, inputs=self.aspect_ratios_selection, outputs=[self.width, self.height])

                        elif category == "cfg":
                            with gr.Row():
                                self.cfg_scale = gr.Slider(minimum=1.0, maximum=30.0, step=0.5, label='CFG Scale', info="The weight of the text prompt, the higher the value the stronger the effect", value=4.5, elem_id=f"{self.id_part}_cfg_scale")

                        elif category == "checkboxes":
                            with FormRow(elem_classes="checkboxes-row", variant="compact"):
                                pass

                        elif category == "accordions":
                            with gr.Row(elem_id=f"{self.id_part}_accordions", elem_classes="accordions"):
                                if not is_img2img:
                                    with InputAccordion(False, label="Hires. fix", elem_id=f"{self.id_part}_hr", visible=True) as self.enable_hr:
                                        with self.enable_hr.extra():
                                            self.hr_final_resolution = FormHTML(value="", elem_id="txtimg_hr_finalres", label="Upscaled resolution", interactive=False, min_width=0)

                                        with FormRow(elem_id=f"{self.id_part}_hires_fix_row1", variant="compact"):
                                            self.hr_upscaler = gr.Dropdown(label="Upscaler", elem_id=f"{self.id_part}_hr_upscaler", choices=[*shared.latent_upscale_modes, *[x.name for x in shared.sd_upscalers]], value=shared.latent_upscale_default_mode)
                                            self.hr_second_pass_steps = gr.Slider(minimum=0, maximum=150, step=1, label='Hires steps', value=0, elem_id=f"{self.id_part}_hires_steps")
                                            self.denoising_strength = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label='Denoising strength', value=0.7, elem_id=f"{self.id_part}_denoising_strength")

                                        with FormRow(elem_id=f"{self.id_part}_hires_fix_row2", variant="compact"):
                                            self.hr_scale = gr.Slider(minimum=1.0, maximum=4.0, step=0.05, label="Upscale by", value=2.0, elem_id=f"{self.id_part}_hr_scale")
                                            self.hr_resize_x = gr.Slider(minimum=0, maximum=2048, step=8, label="Resize width to", value=0, elem_id=f"{self.id_part}_hr_resize_x")
                                            self.hr_resize_y = gr.Slider(minimum=0, maximum=2048, step=8, label="Resize height to", value=0, elem_id=f"{self.id_part}_hr_resize_y")

                                        with FormRow(elem_id=f"{self.id_part}_hires_fix_row3", variant="compact", visible=opts.hires_fix_show_sampler) as self.hr_sampler_container:
                                            self.hr_checkpoint_name = gr.Dropdown(label='Hires checkpoint', elem_id="hr_checkpoint", choices=["Use same checkpoint"] + modules.sd_models.checkpoint_tiles(use_short=True), value="Use same checkpoint")
                                            create_refresh_button(self.hr_checkpoint_name, modules.sd_models.list_models, lambda: {"choices": ["Use same checkpoint"] + modules.sd_models.checkpoint_tiles(use_short=True)}, "hr_checkpoint_refresh")

                                            self.hr_sampler_name = gr.Dropdown(label='Hires sampling method', elem_id="hr_sampler", choices=["Use same sampler"] + sd_samplers.visible_sampler_names(), value="Use same sampler")

                                        with FormRow(elem_id=f"{self.id_part}_hires_fix_row4", variant="compact", visible=opts.hires_fix_show_prompts) as self.hr_prompts_container:
                                            with gr.Column(scale=80):
                                                with gr.Row():
                                                    self.hr_prompt = gr.Textbox(label="Hires prompt", elem_id="hires_prompt", show_label=False, lines=3, placeholder="Prompt for hires fix pass.\nLeave empty to use the same prompt as in first pass.", elem_classes=["prompt"])
                                            with gr.Column(scale=80):
                                                with gr.Row():
                                                    self.hr_negative_prompt = gr.Textbox(label="Hires negative prompt", elem_id="hires_neg_prompt", show_label=False, lines=3, placeholder="Negative prompt for hires fix pass.\nLeave empty to use the same negative prompt as in first pass.", elem_classes=["prompt"])
                                script_runner.setup_ui_for_section(category)

                        elif category == "batch":
                            if not opts.dimensions_and_batch_together:
                                with FormRow(elem_id=f"{self.id_part}_column_batch"):
                                    self.batch_count = gr.Slider(minimum=1, maximum=32, step=1, label='Batch count', value=1, elem_id=f"{self.id_part}_batch_count")
                                    self.batch_size = gr.Slider(visible=False, minimum=1, maximum=8, step=1, label='Batch size', value=1, elem_id=f"{self.id_part}_batch_size")

                        elif category == "override_settings":
                            with FormRow(elem_id=f"{self.id_part}_override_settings_row") as row:
                                self.override_settings = create_override_settings_dropdown('txt2img', row)

                        elif category == "scripts":
                            pass

                        if category not in {"accordions"}:
                            script_runner = getattr(scripts, f"scripts_{self.id_part}")
                            script_runner.setup_ui_for_section(category)

                if not is_img2img:
                    hr_resolution_preview_inputs = [self.enable_hr, self.width, self.height, self.hr_scale, self.hr_resize_x, self.hr_resize_y]
                    for component in hr_resolution_preview_inputs:
                        event = component.release if isinstance(component, gr.Slider) else component.change

                        event(
                            fn=calc_resolution_hires,
                            inputs=hr_resolution_preview_inputs,
                            outputs=[self.hr_final_resolution],
                            show_progress=False,
                        )
                        event(
                            None,
                            _js="onCalcResolutionHires",
                            inputs=hr_resolution_preview_inputs,
                            outputs=[],
                            show_progress=False,
                        )

            extra_model_unrelated_tabs = [configuration_tab]
            extra_tabs = {
                "ImagePrompt": "ControlNet",
                "Style": "Style Selector for SDXL 1.0",
            }
            ignored_scripts = set(extra_tabs.values()) | {"Outpainting mk2"}
            with gr.Tab("Extentions", render=False) as extentions_tab:
                with FormGroup(elem_id=f"{self.id_part}_script_container"):
                    self.custom_inputs = scripts_runner.setup_ui(ignored_scripts=ignored_scripts)
            extra_model_unrelated_tabs.append(extentions_tab)

            for tab_name, tab_key in extra_tabs.items():
                # # Disable ImagePrompt for img2img
                # if tab_name == "ImagePrompt" and is_img2img:
                #     continue
                extra_script = scripts_runner.title_map.get(tab_key.lower())
                if extra_script is not None:
                    with gr.Tab(tab_name) as extra_tab:
                        scripts_runner.create_script_ui(extra_script)
                        extra_model_unrelated_tabs.append(extra_tab)


            # with gr.Tabs(visible=False):
            extentions_tab.render()

            with gr.Tab("Extra Models") as extra_model_tab:
                extra_networks_ui = ui_extra_networks.create_ui(interface, extra_model_unrelated_tabs, self.id_part, related_tabs=[extra_model_tab])
                ui_extra_networks.setup_ui(extra_networks_ui, gallery)


class Img2ImgColumn:
    def __init__(self, dummy_component):
        with gr.Column(variant='compact', elem_id="img2img_settings"):
            copy_image_buttons = []
            copy_image_destinations = {}

            def add_copy_image_controls(tab_name, elem):
                with gr.Row(variant="compact", elem_id=f"img2img_copy_to_{tab_name}", visible=False):
                    gr.HTML("Copy image to: ", elem_id=f"img2img_label_copy_to_{tab_name}")

                    for title, name in zip(['Img2img', 'Inpaint'], ['img2img', 'inpaint']):
                        if name == tab_name:
                            gr.Button(title, interactive=False)
                            copy_image_destinations[name] = elem
                            continue

                        button = gr.Button(title)
                        copy_image_buttons.append((button, name, elem))

            with gr.Tabs(elem_id="mode_img2img"):
                img2img_selected_tab = gr.State(0)

                with gr.TabItem('Img2img', id='img2img', elem_id="img2img_img2img_tab") as tab_img2img:
                    self.init_img = gr.Image(label="Image for img2img", elem_id="img2img_image", show_label=False, source="upload", interactive=True, type="pil", tool="editor", image_mode="RGBA", height=opts.img2img_editor_height)
                    add_copy_image_controls('img2img', self.init_img)

                with gr.TabItem('Draw', id='img2img_sketch', elem_id="img2img_img2img_sketch_tab") as tab_sketch:
                    self.sketch = gr.Image(label="Image for img2img", elem_id="img2img_sketch", show_label=False, source="upload", interactive=True, type="pil", tool="color-sketch", image_mode="RGB", height=opts.img2img_editor_height, brush_color=opts.img2img_sketch_default_brush_color)
                    add_copy_image_controls('sketch', self.sketch)

                with gr.TabItem('Inpaint', id='inpaint', elem_id="img2img_inpaint_tab") as tab_inpaint:
                    self.init_img_with_mask = gr.Image(label="Image for inpainting with mask", show_label=False, elem_id="img2maskimg", source="upload", interactive=True, type="pil", tool="sketch", image_mode="RGBA", height=opts.img2img_editor_height, brush_color=opts.img2img_inpaint_mask_brush_color)
                    add_copy_image_controls('inpaint', self.init_img_with_mask)

                with gr.Tabs(visible=False):
                    with gr.TabItem('Inpaint draw', id='inpaint_sketch', elem_id="img2img_inpaint_sketch_tab") as tab_inpaint_color:
                        self.inpaint_color_sketch = gr.Image(label="Color sketch inpainting", show_label=False, elem_id="inpaint_sketch", source="upload", interactive=True, type="pil", tool="color-sketch", image_mode="RGB", height=opts.img2img_editor_height, brush_color=opts.img2img_inpaint_sketch_default_brush_color)
                        self.inpaint_color_sketch_orig = gr.State(None)
                        add_copy_image_controls('inpaint_sketch', self.inpaint_color_sketch)

                        def update_orig(image, state):
                            if image is not None:
                                same_size = state is not None and state.size == image.size
                                has_exact_match = np.any(np.all(np.array(image) == np.array(state), axis=-1))
                                edited = same_size and has_exact_match
                                return image if not edited or state is None else state

                        self.inpaint_color_sketch.change(update_orig, [self.inpaint_color_sketch, self.inpaint_color_sketch_orig], self.inpaint_color_sketch_orig)

                with gr.TabItem('Inpaint upload', id='inpaint_upload', elem_id="img2img_inpaint_upload_tab") as tab_inpaint_upload:
                    self.init_img_inpaint = gr.Image(label="Image for img2img", show_label=False, source="upload", interactive=True, type="pil", elem_id="img_inpaint_base")
                    self.init_mask_inpaint = gr.Image(label="Mask", source="upload", interactive=True, type="pil", image_mode="RGBA", elem_id="img_inpaint_mask")

                with gr.Tabs(visible=False):
                    with gr.TabItem('Batch', id='batch', elem_id="img2img_batch_tab") as tab_batch:
                        hidden = '<br>Disabled when launched with --hide-ui-dir-config.' if shared.cmd_opts.hide_ui_dir_config else ''
                        gr.HTML(
                            "<p style='padding-bottom: 1em;' class=\"text-gray-500\">Process images in a directory on the same machine where the server is running." +
                            "<br>Use an empty output directory to save pictures normally instead of writing to the output directory." +
                            f"<br>Add inpaint batch mask directory to enable inpaint batch processing."
                            f"{hidden}</p>"
                        )
                        self.img2img_batch_input_dir = gr.Textbox(label="Input directory", **shared.hide_dirs, elem_id="img2img_batch_input_dir")
                        self.img2img_batch_output_dir = gr.Textbox(label="Output directory", **shared.hide_dirs, elem_id="img2img_batch_output_dir")
                        self.img2img_batch_inpaint_mask_dir = gr.Textbox(label="Inpaint batch mask directory (required for inpaint batch processing only)", **shared.hide_dirs, elem_id="img2img_batch_inpaint_mask_dir")
                        with gr.Accordion("PNG info", open=False):
                            self.img2img_batch_use_png_info = gr.Checkbox(label="Append png info to prompts", **shared.hide_dirs, elem_id="img2img_batch_use_png_info")
                            self.img2img_batch_png_info_dir = gr.Textbox(label="PNG info directory", **shared.hide_dirs, placeholder="Leave empty to use input directory", elem_id="img2img_batch_png_info_dir")
                            self.img2img_batch_png_info_props = gr.CheckboxGroup(["Prompt", "Negative prompt", "Seed", "CFG scale", "Sampler", "Steps", "Model hash"], label="Parameters to take from png info", info="Prompts from png info will be appended to prompts set in ui.")

                img2img_tabs = [tab_img2img, tab_sketch, tab_inpaint, tab_inpaint_color, tab_inpaint_upload, tab_batch]

                for i, tab in enumerate(img2img_tabs):
                    tab.select(fn=lambda tabnum=i: tabnum, inputs=[], outputs=[img2img_selected_tab])

            def copy_image(img):
                if isinstance(img, dict) and 'image' in img:
                    return img['image']

                return img

            for button, name, elem in copy_image_buttons:
                button: gr.Button
                button.click(
                    fn=lambda: None,
                    _js=f"switch_to_{name.replace(' ', '_')}",
                    inputs=[],
                    outputs=[],
                ).then(
                    fn=copy_image,
                    inputs=[elem],
                    outputs=[copy_image_destinations[name]],
                )

            ordered_uis = list(ordered_ui_categories())
            if "denoising" in ordered_uis:
                ordered_uis.pop(ordered_uis.index("denoising"))
                ordered_uis.insert(0, "denoising")

            for category in ordered_uis:
                if category == "denoising":
                    self.denoising_strength = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label='Variation', info="The higher the variation, the greater the deviation from the original input.", value=0.75, elem_id="img2img_denoising_strength")

                elif category == "cfg":
                    with gr.Row():
                        self.image_cfg_scale = gr.Slider(minimum=0, maximum=3.0, step=0.05, label='Image CFG Scale', value=1.5, elem_id="img2img_image_cfg_scale", visible=False)

                elif category == "inpaint":
                    # with FormGroup(elem_id="inpaint_controls", visible=False) as inpaint_controls:
                    with gr.Accordion(label="Inpaint options", elem_id="inpaint_controls", open=True, visible=False) as inpaint_controls:
                        with FormRow(visible=False):
                            self.mask_blur = gr.Slider(label='Mask blur', info="Blur the mask with Gaussion kernel", minimum=0, maximum=64, step=1, value=4, elem_id="img2img_mask_blur")
                            self.mask_alpha = gr.Slider(label="Mask transparency", visible=False, elem_id="img2img_mask_alpha")

                        with FormRow():
                            self.inpainting_mask_invert = gr.Radio(label='Mask mode', info="Change the area masked or not masked", choices=['Inpaint masked', 'Inpaint not masked'], value='Inpaint masked', type="index", elem_id="img2img_mask_mode")

                        with FormRow():
                            # choices = ['fill', 'original', 'latent noise', 'latent nothing']
                            choices = ['fill', 'original']
                            self.inpainting_fill = gr.Radio(label='Masked content', info="Erase the masked area (fill) or keep it during inference (original)", choices=choices, value='original', type="index", elem_id="img2img_inpainting_fill")

                        with FormRow():
                            with gr.Column():
                                self.inpaint_full_res = gr.Radio(label="Inpaint area", info="Inference the whole image or only the masked area", choices=["Whole picture", "Only masked"], type="index", value="Whole picture", elem_id="img2img_inpaint_full_res")

                            with gr.Column(scale=4):
                                self.inpaint_full_res_padding = gr.Slider(label='Only masked padding, pixels', info="Padding when inference only the masked area", minimum=0, maximum=256, step=4, value=32, elem_id="img2img_inpaint_full_res_padding")

                        with FormRow(visible=True):
                            self.inpainting_method = gr.Radio(label='Inpainting method', choices=["Original", "Finetuned"], value='Finetuned', elem_id="img2img_inpainting_method")

                elif category == "scripts":
                    with gr.Column():
                        with gr.Accordion("Outpaint", visible=False, open=False) as outpainting_controls:
                            key = "Outpainting mk2"
                            extra_script = scripts.scripts_img2img.title_map.get(key.lower())
                            with gr.Column():
                                self.outpainting_enabled = gr.Checkbox(label="Enable outpaint", value=False)
                                scripts.scripts_img2img.create_script_ui(extra_script)

            def select_img2img_tab(tab):
                return gr.update(visible=tab in [2, 3, 4]), gr.update(visible=tab in [2, 3, 4]), gr.update(visible=tab == 3),

            for i, elem in enumerate(img2img_tabs):
                elem.select(
                    fn=lambda tab=i: select_img2img_tab(tab),
                    inputs=[],
                    outputs=[inpaint_controls, outpainting_controls, self.mask_alpha],
                )


class Toprow:
    """Creates a top row UI with prompts, generate button, styles, extra little buttons for things, and enables some functionality related to their operation"""

    def __init__(self, is_img2img):
        id_part = "img2img" if is_img2img else "txt2img"
        self.id_part = id_part

        with gr.Row(elem_id=f"{id_part}_toprow", variant="compact"):
            with gr.Column(elem_id=f"{id_part}_prompt_container", scale=6):
                with gr.Row():
                    with gr.Column(scale=80):
                        with gr.Row():
                            self.prompt = gr.Textbox(label="Prompt", elem_id=f"{id_part}_prompt", show_label=False, lines=3, placeholder="Prompt (press Ctrl+Enter or Alt+Enter to generate)", elem_classes=["prompt"])
                            self.prompt_img = gr.File(label="", elem_id=f"{id_part}_prompt_image", file_count="single", type="binary", visible=False)

                with gr.Row():
                    with gr.Column(scale=80):
                        with gr.Row():
                            self.negative_prompt = gr.Textbox(label="Negative prompt", elem_id=f"{id_part}_neg_prompt", show_label=False, lines=3, placeholder="Negative prompt (press Ctrl+Enter or Alt+Enter to generate)", elem_classes=["prompt"])

            self.button_interrogate = None
            self.button_deepbooru = None
            if is_img2img:
                with gr.Column(scale=1, elem_classes="interrogate-col"):
                    self.button_interrogate = gr.Button('Interrogate\nCLIP', elem_id="interrogate")
                    self.button_deepbooru = gr.Button('Interrogate\nDeepBooru', elem_id="deepbooru")

            with gr.Column(scale=1, elem_id=f"{id_part}_actions_column"):
                with gr.Row(elem_id=f"{id_part}_generate_box", elem_classes="generate-box"):
                    self.interrupt = gr.Button('Interrupt', elem_id=f"{id_part}_interrupt", elem_classes="generate-box-interrupt")
                    self.skip = gr.Button('Skip', elem_id=f"{id_part}_skip", elem_classes="generate-box-skip")
                    self.submit = gr.Button('Generate', elem_id=f"{id_part}_generate", variant='primary')

                    self.skip.click(
                        fn=lambda: shared.state.skip(),
                        inputs=[],
                        outputs=[],
                    )

                    self.interrupt.click(
                        fn=lambda: shared.state.interrupt(),
                        inputs=[],
                        outputs=[],
                    )

                with gr.Row(elem_id=f"{id_part}_tools"):
                    self.paste = ToolButton(value=paste_symbol, elem_id="paste", tooltip="Read generation parameters from prompt or last generation if prompt is empty into user interface.")
                    self.clear_prompt_button = ToolButton(value=clear_prompt_symbol, elem_id=f"{id_part}_clear_prompt", tooltip="Clear prompt")
                    self.apply_styles = ToolButton(value=ui_prompt_styles.styles_materialize_symbol, elem_id=f"{id_part}_style_apply", tooltip="Apply all selected styles to prompts.")
                    self.restore_progress_button = ToolButton(value=restore_progress_symbol, elem_id=f"{id_part}_restore_progress", visible=False, tooltip="Restore progress")

                    self.token_counter = gr.HTML(value="<span>0/75</span>", elem_id=f"{id_part}_token_counter", elem_classes=["token-counter"])
                    self.token_button = gr.Button(visible=False, elem_id=f"{id_part}_token_button")
                    self.negative_token_counter = gr.HTML(value="<span>0/75</span>", elem_id=f"{id_part}_negative_token_counter", elem_classes=["token-counter"])
                    self.negative_token_button = gr.Button(visible=False, elem_id=f"{id_part}_negative_token_button")

                    self.clear_prompt_button.click(
                        fn=lambda *x: x,
                        _js="confirm_clear_prompt",
                        inputs=[self.prompt, self.negative_prompt],
                        outputs=[self.prompt, self.negative_prompt],
                    )

                self.ui_styles = ui_prompt_styles.UiPromptStyles(id_part, self.prompt, self.negative_prompt)
                self.ui_styles.setup_apply_button(self.apply_styles)

        self.prompt_img.change(
            fn=modules.images.image_data,
            inputs=[self.prompt_img],
            outputs=[self.prompt, self.prompt_img],
            show_progress=False,
        )


def setup_progressbar(*args, **kwargs):
    pass


def apply_setting(key, value):
    if value is None:
        return gr.update()

    if shared.cmd_opts.freeze_settings:
        return gr.update()

    # dont allow model to be swapped when model hash exists in prompt
    if key == "sd_model_checkpoint" and opts.disable_weights_auto_swap:
        return gr.update()

    if key == "sd_model_checkpoint":
        ckpt_info = sd_models.get_closet_checkpoint_match(value)

        if ckpt_info is not None:
            value = ckpt_info.title
        else:
            return gr.update()

    comp_args = opts.data_labels[key].component_args
    if comp_args and isinstance(comp_args, dict) and comp_args.get('visible') is False:
        return

    valtype = type(opts.data_labels[key].default)
    oldval = opts.data.get(key, None)
    opts.data[key] = valtype(value) if valtype != type(None) else value
    if oldval != value and opts.data_labels[key].onchange is not None:
        opts.data_labels[key].onchange()

    opts.save(shared.config_filename)
    return getattr(opts, key)


def create_output_panel(tabname, outdir):
    return ui_common.create_output_panel(tabname, outdir)


def create_sampler_and_steps_selection(choices, tabname, sampler_interactive:bool = True):
    if opts.samplers_in_dropdown:
        with FormRow(elem_id=f"sampler_selection_{tabname}"):
            sampler_name = gr.Dropdown(label='Sampling method', elem_id=f"{tabname}_sampling", choices=choices, value=choices[0], interactive=sampler_interactive)
            steps = gr.Slider(minimum=1, maximum=150, step=1, elem_id=f"{tabname}_steps", label="Sampling steps", value=20)
    else:
        with FormGroup(elem_id=f"sampler_selection_{tabname}"):
            steps = gr.Slider(minimum=1, maximum=150, step=1, elem_id=f"{tabname}_steps", label="Sampling steps", value=20)
            sampler_name = gr.Radio(label='Sampling method', elem_id=f"{tabname}_sampling", choices=choices, value=choices[0], interactive=sampler_interactive)

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
    import modules.img2img
    import modules.txt2img

    reload_javascript()

    parameters_copypaste.reset()

    scripts.scripts_current = scripts.scripts_txt2img
    scripts.scripts_txt2img.initialize_scripts(is_img2img=False)

    with gr.Blocks(analytics_enabled=False) as txt2img_interface:
        scripts.scripts_txt2img.prepare_ui()
        dummy_component = gr.Label(visible=False)
        with gr.Row():
            with gr.Column(scale=2, label="Input & Output", elem_id="txt2img_input_and_output"):
                prompt_row = PromptColumn(is_img2img=False)

                with gr.Row(variant="compact"):
                    default_advanced_checkbox = False
                    advanced_checkbox = gr.Checkbox(
                        label="Advanced",
                        value=default_advanced_checkbox,
                        elem_classes="min_check",
                        elem_id="txt2img_advanced_checkbox",
                    )

                # Output
                txt2img_gallery, generation_info, html_info, html_log = ui_common.create_output_panel("txt2img", opts.outdir_txt2img_samples)

            with gr.Column(scale=1, visible=default_advanced_checkbox) as advanced_column:
                advanced_ui = AdvancedColumn(is_img2img=False, interface=txt2img_interface, gallery=txt2img_gallery)

            advanced_checkbox.change(
                lambda x: gr.update(visible=x),
                advanced_checkbox,
                advanced_column,
                queue=False,
            ).then(fn=lambda: None, _js="refresh_grid_delayed", queue=False)

            # Dummy bottons
            dropdown = gr.Dropdown(visible=False, label="Styles", show_label=False, elem_id="txt2img_styles", choices=list(shared.prompt_styles.styles), value=[], multiselect=True, tooltip="Styles")

            txt2img_args = dict(
                fn=wrap_gradio_gpu_call(modules.txt2img.txt2img, extra_outputs=[None, '', '']),
                _js="submit",
                inputs=[
                    dummy_component,
                    prompt_row.prompt,
                    advanced_ui.negative_prompt,
                    dropdown,
                    advanced_ui.steps,
                    advanced_ui.sampler_name,
                    advanced_ui.batch_count,
                    advanced_ui.batch_size,
                    advanced_ui.cfg_scale,
                    advanced_ui.height,
                    advanced_ui.width,
                    advanced_ui.enable_hr,
                    advanced_ui.denoising_strength,
                    advanced_ui.hr_scale,
                    advanced_ui.hr_upscaler,
                    advanced_ui.hr_second_pass_steps,
                    advanced_ui.hr_resize_x,
                    advanced_ui.hr_resize_y,
                    advanced_ui.hr_checkpoint_name,
                    advanced_ui.hr_sampler_name,
                    advanced_ui.hr_prompt,
                    advanced_ui.hr_negative_prompt,
                    advanced_ui.override_settings,

                ] + advanced_ui.custom_inputs,

                outputs=[
                    txt2img_gallery,
                    generation_info,
                    html_info,
                    html_log,
                ],
                show_progress=False,
            )

            prompt_row.prompt.submit(**txt2img_args)
            prompt_row.submit.click(**txt2img_args)

            advanced_ui.res_switch_btn.click(fn=None, _js="function(){switchWidthHeight('txt2img')}", inputs=None, outputs=None, show_progress=False)

            txt2img_paste_fields = [
                (prompt_row.prompt, "Prompt"),
                (advanced_ui.negative_prompt, "Negative prompt"),
                (advanced_ui.steps, "Steps"),
                (advanced_ui.sampler_name, "Sampler"),
                (advanced_ui.cfg_scale, "CFG scale"),
                (advanced_ui.width, "Size-1"),
                (advanced_ui.height, "Size-2"),
                (advanced_ui.batch_size, "Batch size"),
                (dropdown, lambda d: d["Styles array"] if isinstance(d.get("Styles array"), list) else gr.update()),
                (advanced_ui.denoising_strength, "Denoising strength"),
                (advanced_ui.enable_hr, lambda d: "Denoising strength" in d and ("Hires upscale" in d or "Hires upscaler" in d or "Hires resize-1" in d)),
                (advanced_ui.hr_scale, "Hires upscale"),
                (advanced_ui.hr_upscaler, "Hires upscaler"),
                (advanced_ui.hr_second_pass_steps, "Hires steps"),
                (advanced_ui.hr_resize_x, "Hires resize-1"),
                (advanced_ui.hr_resize_y, "Hires resize-2"),
                (advanced_ui.hr_checkpoint_name, "Hires checkpoint"),
                (advanced_ui.hr_sampler_name, "Hires sampler"),
                (advanced_ui.hr_sampler_container, lambda d: gr.update(visible=True) if d.get("Hires sampler", "Use same sampler") != "Use same sampler" or d.get("Hires checkpoint", "Use same checkpoint") != "Use same checkpoint" else gr.update()),
                (advanced_ui.hr_prompt, "Hires prompt"),
                (advanced_ui.hr_negative_prompt, "Hires negative prompt"),
                (advanced_ui.hr_prompts_container, lambda d: gr.update(visible=True) if d.get("Hires prompt", "") != "" or d.get("Hires negative prompt", "") != "" else gr.update()),
                *scripts.scripts_txt2img.infotext_fields
            ]
            parameters_copypaste.add_paste_fields("txt2img", None, txt2img_paste_fields, advanced_ui.override_settings)
            parameters_copypaste.register_paste_params_button(parameters_copypaste.ParamBinding(
                paste_button=advanced_ui.paste, tabname="txt2img", source_text_component=prompt_row.prompt, source_image_component=None,
            ))

            txt2img_preview_params = [
                prompt_row.prompt,
                advanced_ui.negative_prompt,
                advanced_ui.steps,
                advanced_ui.sampler_name,
                advanced_ui.cfg_scale,
                scripts.scripts_txt2img.script('Seed').seed,
                advanced_ui.width,
                advanced_ui.height,
            ]

    scripts.scripts_current = scripts.scripts_img2img
    scripts.scripts_img2img.initialize_scripts(is_img2img=True)

    with gr.Blocks(analytics_enabled=False) as img2img_interface:
        scripts.scripts_img2img.prepare_ui()
        with gr.Row():
            with gr.Column(scale=2, label="Input & Output", elem_id="img2img_input_and_output", render=False) as img2img_io_column:
                prompt_row = PromptColumn(is_img2img=True)
                # Output
                img2img_gallery, generation_info, html_info, html_log = create_output_panel("img2img", opts.outdir_img2img_samples)

            with gr.Column(scale=2, label="Image2Image Column"):
                img2img_column = Img2ImgColumn(dummy_component)

                default_advanced_checkbox = False
                with gr.Accordion(label="Advanced options", open=default_advanced_checkbox) as advanced_column:
                    advanced_ui = AdvancedColumn(is_img2img=True, interface=img2img_interface, gallery=img2img_gallery)

            img2img_io_column.render()

        # Dummy bottons
        scale_by = gr.Slider(visible=False, minimum=0.05, maximum=4.0, step=0.05, label="Scale", value=1.0, elem_id="img2img_scale")
        selected_scale_tab = gr.State(value=0)

        img2img_args = dict(
            fn=wrap_gradio_gpu_call(modules.img2img.img2img, extra_outputs=[None, '', '']),
            _js="submit_img2img",
            inputs=[
                dummy_component,
                dummy_component,
                img2img_column.inpainting_method,
                img2img_column.outpainting_enabled,
                prompt_row.prompt,
                advanced_ui.negative_prompt,
                dropdown,
                img2img_column.init_img,
                img2img_column.sketch,
                img2img_column.init_img_with_mask,
                img2img_column.inpaint_color_sketch,
                img2img_column.inpaint_color_sketch_orig,
                img2img_column.init_img_inpaint,
                img2img_column.init_mask_inpaint,
                advanced_ui.steps,
                advanced_ui.sampler_name,
                img2img_column.mask_blur,
                img2img_column.mask_alpha,
                img2img_column.inpainting_fill,
                advanced_ui.batch_count,
                advanced_ui.batch_size,
                advanced_ui.cfg_scale,
                img2img_column.image_cfg_scale,
                img2img_column.denoising_strength,
                selected_scale_tab,
                advanced_ui.height,
                advanced_ui.width,
                scale_by,
                advanced_ui.resize_mode,
                img2img_column.inpaint_full_res,
                img2img_column.inpaint_full_res_padding,
                img2img_column.inpainting_mask_invert,
                img2img_column.img2img_batch_input_dir,
                img2img_column.img2img_batch_output_dir,
                img2img_column.img2img_batch_inpaint_mask_dir,
                advanced_ui.override_settings,
                img2img_column.img2img_batch_use_png_info,
                img2img_column.img2img_batch_png_info_props,
                img2img_column.img2img_batch_png_info_dir,
            ] + advanced_ui.custom_inputs,
            outputs=[
                img2img_gallery,
                generation_info,
                html_info,
                html_log,
            ],
            show_progress=False,
        )

        prompt_row.prompt.submit(**img2img_args)
        prompt_row.submit.click(**img2img_args)

        def update_image_size(w, h, _):
            downscale = 16
            if w:
                w = int(w // downscale) * downscale
            if h:
                h = int(h // downscale) * downscale
            return w or gr.update(), h or gr.update()

        advanced_ui.res_switch_btn.click(fn=None, _js="function(){switchWidthHeight('img2img')}", inputs=None, outputs=None, show_progress=False)
        advanced_ui.detect_image_size_btn.click(
            fn=update_image_size,
            _js="currentImg2imgSourceResolution",
            inputs=[dummy_component, dummy_component, dummy_component],
            outputs=[advanced_ui.width, advanced_ui.height],
            show_progress=False,
        )

        img2img_paste_fields = [
            (prompt_row.prompt, "Prompt"),
            (advanced_ui.negative_prompt, "Negative prompt"),
            (advanced_ui.steps, "Steps"),
            (advanced_ui.sampler_name, "Sampler"),
            (advanced_ui.cfg_scale, "CFG scale"),
            (img2img_column.image_cfg_scale, "Image CFG scale"),
            (advanced_ui.width, "Size-1"),
            (advanced_ui.height, "Size-2"),
            (advanced_ui.batch_size, "Batch size"),
            (dropdown, lambda d: d["Styles array"] if isinstance(d.get("Styles array"), list) else gr.update()),
            (img2img_column.denoising_strength, "Denoising strength"),
            (img2img_column.mask_blur, "Mask blur"),
            *scripts.scripts_img2img.infotext_fields
        ]
        parameters_copypaste.add_paste_fields("img2img", img2img_column.init_img, img2img_paste_fields, advanced_ui.override_settings)
        parameters_copypaste.add_paste_fields("inpaint", img2img_column.init_img_with_mask, img2img_paste_fields, advanced_ui.override_settings)
        parameters_copypaste.register_paste_params_button(parameters_copypaste.ParamBinding(
            paste_button=advanced_ui.paste, tabname="img2img", source_text_component=prompt_row.prompt, source_image_component=None,
        ))

    scripts.scripts_current = None

    with gr.Blocks(analytics_enabled=False) as extras_interface:
        ui_postprocessing.create_ui()

    with gr.Blocks(analytics_enabled=False) as pnginfo_interface:
        with gr.Row(equal_height=False):
            with gr.Column(variant='panel'):
                image = gr.Image(elem_id="pnginfo_image", label="Source", source="upload", interactive=True, type="pil")

            with gr.Column(variant='panel'):
                html = gr.HTML()
                generation_info = gr.Textbox(visible=False, elem_id="pnginfo_generation_info")
                html2 = gr.HTML()
                with gr.Row():
                    buttons = parameters_copypaste.create_buttons(["txt2img", "img2img", "inpaint", "extras"])

                for tabname, button in buttons.items():
                    parameters_copypaste.register_paste_params_button(parameters_copypaste.ParamBinding(
                        paste_button=button, tabname=tabname, source_text_component=generation_info, source_image_component=image,
                    ))

        image.change(
            fn=wrap_gradio_call(modules.extras.run_pnginfo),
            inputs=[image],
            outputs=[html, generation_info, html2],
        )

    modelmerger_ui = ui_checkpoint_merger.UiCheckpointMerger()

    with gr.Blocks(analytics_enabled=False) as train_interface:
        with gr.Row(equal_height=False):
            gr.HTML(value="<p style='margin-bottom: 0.7em'>See <b><a href=\"https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Textual-Inversion\">wiki</a></b> for detailed explanation.</p>")

        with gr.Row(variant="compact", equal_height=False):
            with gr.Tabs(elem_id="train_tabs"):

                with gr.Tab(label="Create embedding", id="create_embedding"):
                    new_embedding_name = gr.Textbox(label="Name", elem_id="train_new_embedding_name")
                    initialization_text = gr.Textbox(label="Initialization text", value="*", elem_id="train_initialization_text")
                    nvpt = gr.Slider(label="Number of vectors per token", minimum=1, maximum=75, step=1, value=1, elem_id="train_nvpt")
                    overwrite_old_embedding = gr.Checkbox(value=False, label="Overwrite Old Embedding", elem_id="train_overwrite_old_embedding")

                    with gr.Row():
                        with gr.Column(scale=3):
                            gr.HTML(value="")

                        with gr.Column():
                            create_embedding = gr.Button(value="Create embedding", variant='primary', elem_id="train_create_embedding")

                with gr.Tab(label="Create hypernetwork", id="create_hypernetwork"):
                    new_hypernetwork_name = gr.Textbox(label="Name", elem_id="train_new_hypernetwork_name")
                    new_hypernetwork_sizes = gr.CheckboxGroup(label="Modules", value=["768", "320", "640", "1280"], choices=["768", "1024", "320", "640", "1280"], elem_id="train_new_hypernetwork_sizes")
                    new_hypernetwork_layer_structure = gr.Textbox("1, 2, 1", label="Enter hypernetwork layer structure", placeholder="1st and last digit must be 1. ex:'1, 2, 1'", elem_id="train_new_hypernetwork_layer_structure")
                    new_hypernetwork_activation_func = gr.Dropdown(value="linear", label="Select activation function of hypernetwork. Recommended : Swish / Linear(none)", choices=hypernetworks_ui.keys, elem_id="train_new_hypernetwork_activation_func")
                    new_hypernetwork_initialization_option = gr.Dropdown(value = "Normal", label="Select Layer weights initialization. Recommended: Kaiming for relu-like, Xavier for sigmoid-like, Normal otherwise", choices=["Normal", "KaimingUniform", "KaimingNormal", "XavierUniform", "XavierNormal"], elem_id="train_new_hypernetwork_initialization_option")
                    new_hypernetwork_add_layer_norm = gr.Checkbox(label="Add layer normalization", elem_id="train_new_hypernetwork_add_layer_norm")
                    new_hypernetwork_use_dropout = gr.Checkbox(label="Use dropout", elem_id="train_new_hypernetwork_use_dropout")
                    new_hypernetwork_dropout_structure = gr.Textbox("0, 0, 0", label="Enter hypernetwork Dropout structure (or empty). Recommended : 0~0.35 incrementing sequence: 0, 0.05, 0.15", placeholder="1st and last digit must be 0 and values should be between 0 and 1. ex:'0, 0.01, 0'")
                    overwrite_old_hypernetwork = gr.Checkbox(value=False, label="Overwrite Old Hypernetwork", elem_id="train_overwrite_old_hypernetwork")

                    with gr.Row():
                        with gr.Column(scale=3):
                            gr.HTML(value="")

                        with gr.Column():
                            create_hypernetwork = gr.Button(value="Create hypernetwork", variant='primary', elem_id="train_create_hypernetwork")

                with gr.Tab(label="Preprocess images", id="preprocess_images"):
                    process_src = gr.Textbox(label='Source directory', elem_id="train_process_src")
                    process_dst = gr.Textbox(label='Destination directory', elem_id="train_process_dst")
                    process_width = gr.Slider(minimum=64, maximum=2048, step=8, label="Width", value=512, elem_id="train_process_width")
                    process_height = gr.Slider(minimum=64, maximum=2048, step=8, label="Height", value=512, elem_id="train_process_height")
                    preprocess_txt_action = gr.Dropdown(label='Existing Caption txt Action', value="ignore", choices=["ignore", "copy", "prepend", "append"], elem_id="train_preprocess_txt_action")

                    with gr.Row():
                        process_keep_original_size = gr.Checkbox(label='Keep original size', elem_id="train_process_keep_original_size")
                        process_flip = gr.Checkbox(label='Create flipped copies', elem_id="train_process_flip")
                        process_split = gr.Checkbox(label='Split oversized images', elem_id="train_process_split")
                        process_focal_crop = gr.Checkbox(label='Auto focal point crop', elem_id="train_process_focal_crop")
                        process_multicrop = gr.Checkbox(label='Auto-sized crop', elem_id="train_process_multicrop")
                        process_caption = gr.Checkbox(label='Use BLIP for caption', elem_id="train_process_caption")
                        process_caption_deepbooru = gr.Checkbox(label='Use deepbooru for caption', visible=True, elem_id="train_process_caption_deepbooru")

                    with gr.Row(visible=False) as process_split_extra_row:
                        process_split_threshold = gr.Slider(label='Split image threshold', value=0.5, minimum=0.0, maximum=1.0, step=0.05, elem_id="train_process_split_threshold")
                        process_overlap_ratio = gr.Slider(label='Split image overlap ratio', value=0.2, minimum=0.0, maximum=0.9, step=0.05, elem_id="train_process_overlap_ratio")

                    with gr.Row(visible=False) as process_focal_crop_row:
                        process_focal_crop_face_weight = gr.Slider(label='Focal point face weight', value=0.9, minimum=0.0, maximum=1.0, step=0.05, elem_id="train_process_focal_crop_face_weight")
                        process_focal_crop_entropy_weight = gr.Slider(label='Focal point entropy weight', value=0.15, minimum=0.0, maximum=1.0, step=0.05, elem_id="train_process_focal_crop_entropy_weight")
                        process_focal_crop_edges_weight = gr.Slider(label='Focal point edges weight', value=0.5, minimum=0.0, maximum=1.0, step=0.05, elem_id="train_process_focal_crop_edges_weight")
                        process_focal_crop_debug = gr.Checkbox(label='Create debug image', elem_id="train_process_focal_crop_debug")

                    with gr.Column(visible=False) as process_multicrop_col:
                        gr.Markdown('Each image is center-cropped with an automatically chosen width and height.')
                        with gr.Row():
                            process_multicrop_mindim = gr.Slider(minimum=64, maximum=2048, step=8, label="Dimension lower bound", value=384, elem_id="train_process_multicrop_mindim")
                            process_multicrop_maxdim = gr.Slider(minimum=64, maximum=2048, step=8, label="Dimension upper bound", value=768, elem_id="train_process_multicrop_maxdim")
                        with gr.Row():
                            process_multicrop_minarea = gr.Slider(minimum=64*64, maximum=2048*2048, step=1, label="Area lower bound", value=64*64, elem_id="train_process_multicrop_minarea")
                            process_multicrop_maxarea = gr.Slider(minimum=64*64, maximum=2048*2048, step=1, label="Area upper bound", value=640*640, elem_id="train_process_multicrop_maxarea")
                        with gr.Row():
                            process_multicrop_objective = gr.Radio(["Maximize area", "Minimize error"], value="Maximize area", label="Resizing objective", elem_id="train_process_multicrop_objective")
                            process_multicrop_threshold = gr.Slider(minimum=0, maximum=1, step=0.01, label="Error threshold", value=0.1, elem_id="train_process_multicrop_threshold")

                    with gr.Row():
                        with gr.Column(scale=3):
                            gr.HTML(value="")

                        with gr.Column():
                            with gr.Row():
                                interrupt_preprocessing = gr.Button("Interrupt", elem_id="train_interrupt_preprocessing")
                            run_preprocess = gr.Button(value="Preprocess", variant='primary', elem_id="train_run_preprocess")

                    process_split.change(
                        fn=lambda show: gr_show(show),
                        inputs=[process_split],
                        outputs=[process_split_extra_row],
                    )

                    process_focal_crop.change(
                        fn=lambda show: gr_show(show),
                        inputs=[process_focal_crop],
                        outputs=[process_focal_crop_row],
                    )

                    process_multicrop.change(
                        fn=lambda show: gr_show(show),
                        inputs=[process_multicrop],
                        outputs=[process_multicrop_col],
                    )

                def get_textual_inversion_template_names():
                    return sorted(textual_inversion.textual_inversion_templates)

                with gr.Tab(label="Train", id="train"):
                    gr.HTML(value="<p style='margin-bottom: 0.7em'>Train an embedding or Hypernetwork; you must specify a directory with a set of 1:1 ratio images <a href=\"https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Textual-Inversion\" style=\"font-weight:bold;\">[wiki]</a></p>")
                    with FormRow():
                        train_embedding_name = gr.Dropdown(label='Embedding', elem_id="train_embedding", choices=sorted(sd_hijack.model_hijack.embedding_db.word_embeddings.keys()))
                        create_refresh_button(train_embedding_name, sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings, lambda: {"choices": sorted(sd_hijack.model_hijack.embedding_db.word_embeddings.keys())}, "refresh_train_embedding_name")

                        train_hypernetwork_name = gr.Dropdown(label='Hypernetwork', elem_id="train_hypernetwork", choices=sorted(shared.hypernetworks))
                        create_refresh_button(train_hypernetwork_name, shared.reload_hypernetworks, lambda: {"choices": sorted(shared.hypernetworks)}, "refresh_train_hypernetwork_name")

                    with FormRow():
                        embedding_learn_rate = gr.Textbox(label='Embedding Learning rate', placeholder="Embedding Learning rate", value="0.005", elem_id="train_embedding_learn_rate")
                        hypernetwork_learn_rate = gr.Textbox(label='Hypernetwork Learning rate', placeholder="Hypernetwork Learning rate", value="0.00001", elem_id="train_hypernetwork_learn_rate")

                    with FormRow():
                        clip_grad_mode = gr.Dropdown(value="disabled", label="Gradient Clipping", choices=["disabled", "value", "norm"])
                        clip_grad_value = gr.Textbox(placeholder="Gradient clip value", value="0.1", show_label=False)

                    with FormRow():
                        batch_size = gr.Number(visible=False, label='Batch size', value=1, precision=0, elem_id="train_batch_size")
                        gradient_step = gr.Number(label='Gradient accumulation steps', value=1, precision=0, elem_id="train_gradient_step")

                    dataset_directory = gr.Textbox(label='Dataset directory', placeholder="Path to directory with input images", elem_id="train_dataset_directory")
                    log_directory = gr.Textbox(label='Log directory', placeholder="Path to directory where to write outputs", value="textual_inversion", elem_id="train_log_directory")

                    with FormRow():
                        template_file = gr.Dropdown(label='Prompt template', value="style_filewords.txt", elem_id="train_template_file", choices=get_textual_inversion_template_names())
                        create_refresh_button(template_file, textual_inversion.list_textual_inversion_templates, lambda: {"choices": get_textual_inversion_template_names()}, "refrsh_train_template_file")

                    training_width = gr.Slider(minimum=64, maximum=2048, step=8, label="Width", value=512, elem_id="train_training_width")
                    training_height = gr.Slider(minimum=64, maximum=2048, step=8, label="Height", value=512, elem_id="train_training_height")
                    varsize = gr.Checkbox(label="Do not resize images", value=False, elem_id="train_varsize")
                    steps = gr.Number(label='Max steps', value=100000, precision=0, elem_id="train_steps")

                    with FormRow():
                        create_image_every = gr.Number(label='Save an image to log directory every N steps, 0 to disable', value=500, precision=0, elem_id="train_create_image_every")
                        save_embedding_every = gr.Number(label='Save a copy of embedding to log directory every N steps, 0 to disable', value=500, precision=0, elem_id="train_save_embedding_every")

                    use_weight = gr.Checkbox(label="Use PNG alpha channel as loss weight", value=False, elem_id="use_weight")

                    save_image_with_stored_embedding = gr.Checkbox(label='Save images with embedding in PNG chunks', value=True, elem_id="train_save_image_with_stored_embedding")
                    preview_from_txt2img = gr.Checkbox(label='Read parameters (prompt, etc...) from txt2img tab when making previews', value=False, elem_id="train_preview_from_txt2img")

                    shuffle_tags = gr.Checkbox(label="Shuffle tags by ',' when creating prompts.", value=False, elem_id="train_shuffle_tags")
                    tag_drop_out = gr.Slider(minimum=0, maximum=1, step=0.1, label="Drop out tags when creating prompts.", value=0, elem_id="train_tag_drop_out")

                    latent_sampling_method = gr.Radio(label='Choose latent sampling method', value="once", choices=['once', 'deterministic', 'random'], elem_id="train_latent_sampling_method")

                    with gr.Row():
                        train_embedding = gr.Button(value="Train Embedding", variant='primary', elem_id="train_train_embedding")
                        interrupt_training = gr.Button(value="Interrupt", elem_id="train_interrupt_training")
                        train_hypernetwork = gr.Button(value="Train Hypernetwork", variant='primary', elem_id="train_train_hypernetwork")

                params = script_callbacks.UiTrainTabParams(txt2img_preview_params)

                script_callbacks.ui_train_tabs_callback(params)

            with gr.Column(elem_id='ti_gallery_container'):
                ti_output = gr.Text(elem_id="ti_output", value="", show_label=False)
                gr.Gallery(label='Output', show_label=False, elem_id='ti_gallery', columns=4)
                gr.HTML(elem_id="ti_progress", value="")
                ti_outcome = gr.HTML(elem_id="ti_error", value="")

        create_embedding.click(
            fn=textual_inversion_ui.create_embedding,
            inputs=[
                new_embedding_name,
                initialization_text,
                nvpt,
                overwrite_old_embedding,
            ],
            outputs=[
                train_embedding_name,
                ti_output,
                ti_outcome,
            ]
        )

        create_hypernetwork.click(
            fn=hypernetworks_ui.create_hypernetwork,
            inputs=[
                new_hypernetwork_name,
                new_hypernetwork_sizes,
                overwrite_old_hypernetwork,
                new_hypernetwork_layer_structure,
                new_hypernetwork_activation_func,
                new_hypernetwork_initialization_option,
                new_hypernetwork_add_layer_norm,
                new_hypernetwork_use_dropout,
                new_hypernetwork_dropout_structure
            ],
            outputs=[
                train_hypernetwork_name,
                ti_output,
                ti_outcome,
            ]
        )

        run_preprocess.click(
            fn=wrap_gradio_gpu_call(textual_inversion_ui.preprocess, extra_outputs=[gr.update()]),
            _js="start_training_textual_inversion",
            inputs=[
                dummy_component,
                process_src,
                process_dst,
                process_width,
                process_height,
                preprocess_txt_action,
                process_keep_original_size,
                process_flip,
                process_split,
                process_caption,
                process_caption_deepbooru,
                process_split_threshold,
                process_overlap_ratio,
                process_focal_crop,
                process_focal_crop_face_weight,
                process_focal_crop_entropy_weight,
                process_focal_crop_edges_weight,
                process_focal_crop_debug,
                process_multicrop,
                process_multicrop_mindim,
                process_multicrop_maxdim,
                process_multicrop_minarea,
                process_multicrop_maxarea,
                process_multicrop_objective,
                process_multicrop_threshold,
            ],
            outputs=[
                ti_output,
                ti_outcome,
            ],
        )

        train_embedding.click(
            fn=wrap_gradio_gpu_call(textual_inversion_ui.train_embedding, extra_outputs=[gr.update()]),
            _js="start_training_textual_inversion",
            inputs=[
                dummy_component,
                train_embedding_name,
                embedding_learn_rate,
                batch_size,
                gradient_step,
                dataset_directory,
                log_directory,
                training_width,
                training_height,
                varsize,
                steps,
                clip_grad_mode,
                clip_grad_value,
                shuffle_tags,
                tag_drop_out,
                latent_sampling_method,
                use_weight,
                create_image_every,
                save_embedding_every,
                template_file,
                save_image_with_stored_embedding,
                preview_from_txt2img,
                *txt2img_preview_params,
            ],
            outputs=[
                ti_output,
                ti_outcome,
            ]
        )

        train_hypernetwork.click(
            fn=wrap_gradio_gpu_call(hypernetworks_ui.train_hypernetwork, extra_outputs=[gr.update()]),
            _js="start_training_textual_inversion",
            inputs=[
                dummy_component,
                train_hypernetwork_name,
                hypernetwork_learn_rate,
                batch_size,
                gradient_step,
                dataset_directory,
                log_directory,
                training_width,
                training_height,
                varsize,
                steps,
                clip_grad_mode,
                clip_grad_value,
                shuffle_tags,
                tag_drop_out,
                latent_sampling_method,
                use_weight,
                create_image_every,
                save_embedding_every,
                template_file,
                preview_from_txt2img,
                *txt2img_preview_params,
            ],
            outputs=[
                ti_output,
                ti_outcome,
            ]
        )

        interrupt_training.click(
            fn=lambda: shared.state.interrupt(),
            inputs=[],
            outputs=[],
        )

        interrupt_preprocessing.click(
            fn=lambda: shared.state.interrupt(),
            inputs=[],
            outputs=[],
        )

    loadsave = ui_loadsave.UiLoadsave(cmd_opts.ui_config_file)

    settings = ui_settings.UiSettings()
    settings.create_ui(loadsave, dummy_component)

    interfaces = [
        (txt2img_interface, "txt2img", "txt2img"),
        (img2img_interface, "img2img", "img2img"),
        (extras_interface, "Extras", "extras"),
        (pnginfo_interface, "PNG Info", "pnginfo"),
        (modelmerger_ui.blocks, "Checkpoint Merger", "modelmerger"),
        (train_interface, "Train", "train"),
    ]

    interfaces += script_callbacks.ui_tabs_callback()
    interfaces += [(settings.interface, "Settings", "settings")]

    extensions_interface = ui_extensions.create_ui()
    interfaces += [(extensions_interface, "Extensions", "extensions")]

    shared.tab_names = []
    for _interface, label, _ifid in interfaces:
        shared.tab_names.append(label)

    with gr.Blocks(theme=shared.gradio_theme, analytics_enabled=False, title="iFashion AIGC Platform") as demo:
        settings.add_quicksettings()

        parameters_copypaste.connect_paste_params_buttons()

        with gr.Tabs(elem_id="tabs") as tabs:
            tab_order = {k: i for i, k in enumerate(opts.ui_tab_order)}
            sorted_interfaces = sorted(interfaces, key=lambda x: tab_order.get(x[1], 9999))

            for interface, label, ifid in sorted_interfaces:
                if label in shared.opts.hidden_tabs:
                    with gr.Tabs(visible=False):
                        with gr.TabItem(label, id=ifid, elem_id=f"tab_{ifid}"):
                            interface.render()
                else:
                    with gr.TabItem(label, id=ifid, elem_id=f"tab_{ifid}"):
                        interface.render()

                if ifid not in ["extensions", "settings"]:
                    loadsave.add_block(interface, ifid)

            loadsave.add_component(f"webui/Tabs@{tabs.elem_id}", tabs)

            loadsave.setup_ui()

        if os.path.exists(os.path.join(script_path, "notification.mp3")) and shared.opts.notification_audio:
            gr.Audio(interactive=False, value=os.path.join(script_path, "notification.mp3"), elem_id="audio_notification", visible=False)

        footer = shared.html("footer-ifahsion.html")
        footer = footer.format(versions=versions_html(), api_docs="/docs" if shared.cmd_opts.api else "https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/API")
        gr.HTML(footer, elem_id="footer")

        settings.add_functionality(demo)

        update_image_cfg_scale_visibility = lambda: gr.update(visible=shared.sd_model and shared.sd_model.cond_stage_key == "edit")
        settings.text_settings.change(fn=update_image_cfg_scale_visibility, inputs=[], outputs=[img2img_column.image_cfg_scale])
        demo.load(fn=update_image_cfg_scale_visibility, inputs=[], outputs=[img2img_column.image_cfg_scale])

        modelmerger_ui.setup_ui(dummy_component=dummy_component, sd_model_checkpoint_component=settings.component_dict['sd_model_checkpoint'])

        def switch_sd_version(use_sdxl: bool, sdxl_model_checkpoint: str, sdxl_vae: str, sd_model_checkpoint: str, sd_vae: str):
            aspect_ratios = sdxl_aspect_ratios if use_sdxl else sd15_aspect_ratios
            return [
                gr.Dropdown.update(value=sdxl_model_checkpoint if use_sdxl else sd_model_checkpoint),
                gr.Dropdown.update(value=sdxl_vae if use_sdxl else sd_vae),
                gr.Radio.update(choices=aspect_ratios, value=None),
            ]

        SC = shared.settings_components
        SC["sdxl_filter_enabled"].change(switch_sd_version,
            inputs=[SC["sdxl_filter_enabled"], SC["sdxl_default_checkpoint"], SC["sdxl_default_vae"], SC["sd15_default_checkpoint"], SC["sd15_default_vae"]],
            outputs=[SC["sd_model_checkpoint"], SC["sd_vae"], advanced_ui.aspect_ratios_selection], queue=False)

    loadsave.dump_defaults()
    demo.ui_loadsave = loadsave

    return demo


def versions_html():
    import torch
    import launch

    python_version = ".".join([str(x) for x in sys.version_info[0:3]])
    commit = launch.commit_hash()
    tag = launch.git_tag()

    if shared.xformers_available:
        import xformers
        xformers_version = xformers.__version__
    else:
        xformers_version = "N/A"

    """version: <a href="https://github.com/AUTOMATIC1111/stable-diffusion-webui/commit/{commit}">{tag}</a>
&#x2000;•&#x2000;"""
    """&#x2000;•&#x2000;
checkpoint: <a id="sd_checkpoint_hash">N/A</a>"""
    return f"""
python: <span title="{sys.version}">{python_version}</span>
&#x2000;•&#x2000;
torch: {getattr(torch, '__long_version__',torch.__version__)}
&#x2000;•&#x2000;
xformers: {xformers_version}
&#x2000;•&#x2000;
gradio: {gr.__version__}
"""


def setup_ui_api(app):
    from pydantic import BaseModel, Field

    class QuicksettingsHint(BaseModel):
        name: str = Field(title="Name of the quicksettings field")
        label: str = Field(title="Label of the quicksettings field")

    def quicksettings_hint():
        return [QuicksettingsHint(name=k, label=v.label) for k, v in opts.data_labels.items()]

    app.add_api_route("/internal/quicksettings-hint", quicksettings_hint, methods=["GET"], response_model=list[QuicksettingsHint])

    app.add_api_route("/internal/ping", lambda: {}, methods=["GET"])

    app.add_api_route("/internal/profile-startup", lambda: timer.startup_record, methods=["GET"])

    def download_sysinfo(attachment=False):
        from fastapi.responses import PlainTextResponse

        text = sysinfo.get()
        filename = f"sysinfo-{datetime.datetime.utcnow().strftime('%Y-%m-%d-%H-%M')}.txt"

        return PlainTextResponse(text, headers={'Content-Disposition': f'{"attachment" if attachment else "inline"}; filename="{filename}"'})

    app.add_api_route("/internal/sysinfo", download_sysinfo, methods=["GET"])
    app.add_api_route("/internal/sysinfo-download", lambda: download_sysinfo(attachment=True), methods=["GET"])

