import mimetypes
import random
import warnings
from modules import ui_light_html
import modules.gradio_hijack as grh
from modules.ui_gradio_extensions import reload_javascript

import gradio as gr
import gradio.utils
import numpy as np
from PIL import Image, PngImagePlugin  # noqa: F401

from modules.call_queue import wrap_gradio_call, wrap_gradio_gpu_call, wrap_queued_call
from modules.shared import cmd_opts, opts
import modules.shared as shared

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


def create_ui():
    import modules.txt2img

    reload_javascript()

    demo = gr.Blocks(title="iFahsion AIGC Plateform", css=ui_light_html.css).queue()

    with demo:
        with gr.Row():
            with gr.Column(scale=2):
                with gr.Row():
                    progress_window = grh.Image(
                        label="Preview",
                        show_label=True,
                        visible=False,
                        height=768,
                        elem_classes=["main_view"],
                    )
                    progress_gallery = gr.Gallery(
                        label="Finished Images",
                        show_label=True,
                        object_fit="contain",
                        height=768,
                        visible=False,
                        elem_classes=["main_view", "image_gallery"],
                    )
                progress_html = gr.HTML(
                    value=ui_light_html.make_progress_html(32, "Progress 32%"),
                    visible=False,
                    elem_id="progress-bar",
                    elem_classes="progress-bar",
                )
                gallery = gr.Gallery(
                    label="Gallery",
                    show_label=False,
                    object_fit="contain",
                    visible=True,
                    height=768,
                    elem_classes=[
                        "resizable_area",
                        "main_view",
                        "final_gallery",
                        "image_gallery",
                    ],
                    elem_id="final_gallery",
                )
                with gr.Row(elem_classes="type_row"):
                    with gr.Column(scale=17):
                        prompt = gr.Textbox(
                            show_label=False,
                            placeholder="Type prompt here.",
                            elem_id="positive_prompt",
                            container=False,
                            autofocus=True,
                            elem_classes="type_row",
                            lines=1024,
                        )

                        default_prompt = ""
                        if isinstance(default_prompt, str) and default_prompt != "":
                            demo.load(lambda: default_prompt, outputs=prompt)

                    with gr.Column(scale=3, min_width=0):
                        generate_button = gr.Button(
                            label="Generate",
                            value="Generate",
                            elem_classes="type_row",
                            elem_id="generate_button",
                            visible=True,
                        )
                        skip_button = gr.Button(
                            label="Skip",
                            value="Skip",
                            elem_classes="type_row_half",
                            visible=False,
                        )
                        stop_button = gr.Button(
                            label="Stop",
                            value="Stop",
                            elem_classes="type_row_half",
                            elem_id="stop_button",
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

                # TODO: advanced
                default_advanced_checkbox = False
                with gr.Row(elem_classes="advanced_check_row"):
                    advanced_checkbox = gr.Checkbox(
                        label="Advanced",
                        value=default_advanced_checkbox,
                        container=False,
                        elem_classes="min_check",
                    )

            default_image_number = 1
            default_prompt_negative = ""
            with gr.Column(
                scale=1, visible=default_advanced_checkbox
            ) as advanced_column:
                with gr.Tab(label="Setting"):
                    image_number = gr.Slider(
                        label="Image Number",
                        minimum=1,
                        maximum=32,
                        step=1,
                        value=default_image_number,
                    )
                    negative_prompt = gr.Textbox(
                        label="Negative Prompt",
                        show_label=True,
                        placeholder="Type prompt here.",
                        info="Describing what you do not want to see.",
                        lines=2,
                        elem_id="negative_prompt",
                        value=default_prompt_negative,
                    )
                    seed_random = gr.Checkbox(label="Random", value=True)
                    image_seed = gr.Textbox(
                        label="Seed", value=0, max_lines=1, visible=False
                    )  # workaround for https://github.com/gradio-app/gradio/issues/5354

                    def random_checked(r):
                        return gr.update(visible=not r)

                    def refresh_seed(r, seed_string):
                        MIN_SEED = 0
                        MAX_SEED = 2**63 - 1
                        if r:
                            return random.randint(MIN_SEED, MAX_SEED)
                        else:
                            try:
                                seed_value = int(seed_string)
                                if MIN_SEED <= seed_value <= MAX_SEED:
                                    return seed_value
                            except ValueError:
                                pass
                            return random.randint(MIN_SEED, MAX_SEED)

                    seed_random.change(
                        random_checked,
                        inputs=[seed_random],
                        outputs=[image_seed],
                        queue=False,
                    )

            advanced_checkbox.change(
                lambda x: gr.update(visible=x),
                advanced_checkbox,
                advanced_column,
                queue=False,
            ).then(fn=lambda: None, _js="refresh_grid_delayed", queue=False)

            # # TODO:
            # generate_button.click(
            #     lambda: (
            #         gr.update(visible=True, interactive=True),
            #         gr.update(visible=True, interactive=True),
            #         gr.update(visible=False),
            #         [],
            #     ),
            #     outputs=[stop_button, skip_button, generate_button, gallery],
            # ).then(
            #     fn=refresh_seed, inputs=[seed_random, image_seed], outputs=image_seed
            # ).then(
            #     advanced_parameters.set_all_advanced_parameters, inputs=adps
            # ).then(
            #     fn=generate_clicked,
            #     inputs=ctrls,
            #     outputs=[progress_html, progress_window, progress_gallery, gallery],
            # ).then(
            #     lambda: (
            #         gr.update(visible=True),
            #         gr.update(visible=False),
            #         gr.update(visible=False),
            #     ),
            #     outputs=[generate_button, stop_button, skip_button],
            # ).then(
            #     fn=lambda: None, _js="playNotification"
            # ).then(
            #     fn=lambda: None, _js="refresh_grid_delayed"
            # )
    return demo
