import streamlit as st
from generate_shorts import generate_shorts
from transformers import BartTokenizer, BartForConditionalGeneration
from diffusers import AnimateDiffPipeline, MotionAdapter, EulerDiscreteScheduler
from diffusers.utils import export_to_gif
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import torch

# Initialize device and dtype
device = "cuda"
dtype = torch.float16

# Configuration for Animate Diffusion
step = 4  # Options: [1, 2, 4, 8]
repo = "ByteDance/AnimateDiff-Lightning"
ckpt = f"animatediff_lightning_{step}step_diffusers.safetensors"
base = "emilianJR/epiCRealism"  # Choose your favorite base model

# Title and description for your Streamlit app
st.title('Summary-based Image Shorts Generator')
st.markdown("""
            This app generates image shorts based on a provided summary.
            """)

# Button to trigger processing
if st.button('Generate Shorts'):
    try:
        # Read the summary from the file
        st.write("Reading the summary from the file...")
        with open('summary.txt', 'r') as file:
            summary = file.read()
        st.write("Summary read from file")
        st.write(summary)

        # Loading the Animate Diffusion Pipeline
        st.write('Loading the Animate Diffusion Pipeline...')
        adapter = MotionAdapter().to(device, dtype)
        adapter.load_state_dict(load_file(hf_hub_download(repo, ckpt), device=device))

        pipe = AnimateDiffPipeline.from_pretrained(base, motion_adapter=adapter, torch_dtype=dtype).to(device)
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing", beta_schedule="linear")

        # Generating shorts
        st.write('Generating shorts...')
        shorts_path = generate_shorts(summary, pipe, step)
        st.image(str(shorts_path))
    
    except Exception as e:
        st.error(f'Error occurred: {str(e)}')
        st.write(f'Error details: {str(e)}')

else:
    st.write("Waiting for user input")
