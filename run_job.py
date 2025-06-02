import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
import sys
from typing import Union, OrderedDict
from dotenv import load_dotenv
# Load the .env file if it exists
load_dotenv()

sys.path.insert(0, os.getcwd())
# must come before ANY torch or fastai imports
# import toolkit.cuda_malloc

# turn off diffusers telemetry until I can figure out how to make it opt-in
os.environ['DISABLE_TELEMETRY'] = 'YES'

# check if we have DEBUG_TOOLKIT in env
if os.environ.get("DEBUG_TOOLKIT", "0") == "1":
    # set torch to trace mode
    import torch
    torch.autograd.set_detect_anomaly(True)
import argparse
from toolkit.job import get_job
from toolkit.accelerator import get_accelerator
from toolkit.print import print_acc, setup_log_to_file
import pandas as pd

accelerator = get_accelerator()

df=pd.read_csv('/Volumes/dsp_lab/dsp_sandbox_tbls/data/flux_finetuning_input.csv')
          
from collections import OrderedDict

job_to_run = OrderedDict([
    ('job', 'extension'),
    ('config', OrderedDict([
        # this name will be the folder and filename name
        ('name', 'my_first_flux_lora_v1'),
        ('process', [
            OrderedDict([
                ('type', 'sd_trainer'),
                # root folder to save training sessions/samples/weights
                ('training_folder', '/Volumes/dsp_lab/dsp_sandbox_tbls/data/flux_finetuning_output_3'),
                # uncomment to see performance stats in the terminal every N steps
                #('performance_log_every', 1000),
                ('device', 'cuda'),
                # if a trigger word is specified, it will be added to captions of training data if it does not already exist
                # alternatively, in your captions you can add [trigger] and it will be replaced with the trigger word
                # ('trigger_word', 'image'),
                ('network', OrderedDict([
                    ('type', 'lora'),
                    ('linear', 32),
                    ('linear_alpha', 32)
                ])),
                ('save', OrderedDict([
                    ('dtype', 'float16'),  # precision to save
                    ('save_every', 500),  # save every this many steps
                    ('max_step_saves_to_keep', 4)  # how many intermittent saves to keep
                ])),
                ('datasets', [
                    # datasets are a folder of images. captions need to be txt files with the same name as the image
                    # for instance image2.jpg and image2.txt. Only jpg, jpeg, and png are supported currently
                    # images will automatically be resized and bucketed into the resolution specified
                    OrderedDict([
                        ('folder_path', '/Volumes/dsp_lab/dsp_sandbox_tbls/data/ordered_files/'),
                        ('caption_ext', 'txt'),
                        ('caption_dropout_rate', 0.05),  # will drop out the caption 5% of time
                        ('shuffle_tokens', False),  # shuffle caption order, split by commas
                        ('cache_latents_to_disk', True),  # leave this true unless you know what you're doing
                        ('resolution', [1024, 512, 1024])  # flux enjoys multiple resolutions
                    ])
                ]),
                ('train', OrderedDict([
                    ('batch_size', 1),
                    ('steps', 4000),  # total number of steps to train 500 - 4000 is a good range
                    ('gradient_accumulation_steps', 1),
                    ('train_unet', True),
                    ('train_text_encoder', False),  # probably won't work with flux
                    ('content_or_style', 'balanced'),  # content, style, balanced
                    ('gradient_checkpointing', True),  # need the on unless you have a ton of vram
                    ('noise_scheduler', 'flowmatch'),  # for training only
                    ('optimizer', 'adamw8bit'),
                    ('lr', 1e-4),

                    # uncomment this to skip the pre training sample
                    # ('skip_first_sample', True),

                    # uncomment to completely disable sampling
                    # ('disable_sampling', True),

                    # uncomment to use new vell curved weighting. Experimental but may produce better results
                    # ('linear_timesteps', True),

                    # ema will smooth out learning, but could slow it down. Recommended to leave on.
                    ('ema_config', OrderedDict([
                        ('use_ema', True),
                        ('ema_decay', 0.99)
                    ])),

                    # will probably need this if gpu supports it for flux, other dtypes may not work correctly
                    ('dtype', 'bf16')
                ])),
                ('model', OrderedDict([
                    # huggingface model name or path
                    ('name_or_path', 'black-forest-labs/FLUX.1-dev'),
                    ('is_flux', True),
                    ('quantize', False),  # run 8bit mixed precision
                    ('low_vram', True),  # uncomment this if the GPU is connected to your monitors. It will use less vram to quantize, but is slower.
                ])),
                ('sample', OrderedDict([
                    ('sampler', 'flowmatch'),  # must match train.noise_scheduler
                    ('sample_every', 250),  # sample every this many steps
                    ('width', 512),
                    ('height', 512),
                    ('prompts', df.prompt.tolist()),
                    ('neg', ''),  # not used on flux
                    ('seed', 42),
                    ('walk_seed', True),
                    ('guidance_scale', 4),
                    ('sample_steps', 20)
                ]))
            ])
        ])
    ])),
    # you can add any additional meta info here. [name] is replaced with config name at top
    ('meta', OrderedDict([
        ('name', '[name]'),
        ('version', '1.0')
    ]))
])


def print_end_message(jobs_completed, jobs_failed):
    if not accelerator.is_main_process:
        return
    failure_string = f"{jobs_failed} failure{'' if jobs_failed == 1 else 's'}" if jobs_failed > 0 else ""
    completed_string = f"{jobs_completed} completed job{'' if jobs_completed == 1 else 's'}"

    print_acc("")
    print_acc("========================================")
    print_acc("Result:")
    if len(completed_string) > 0:
        print_acc(f" - {completed_string}")
    if len(failure_string) > 0:
        print_acc(f" - {failure_string}")
    print_acc("========================================")


def main():
    
    job = get_job(job_to_run)
    job.run()
    job.cleanup()
    # jobs_completed += 1


if __name__ == '__main__':
    main()
