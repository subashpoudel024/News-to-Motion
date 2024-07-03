import streamlit as st
from scrap import scrap
from summarize import summarize , generate_summary_bart
from generate_shorts import generate_shorts
# from cosine_similarity import cosine_similarity
from transformers import T5ForConditionalGeneration, T5Tokenizer 
import torch
from diffusers import AnimateDiffPipeline, MotionAdapter, EulerDiscreteScheduler
from diffusers.utils import export_to_gif
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from transformers import BartTokenizer, BartForConditionalGeneration


# Initialize BART model and tokenizer
bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
bart_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')


device = "cuda"
dtype = torch.float16

step = 4  # Options: [1,2,4,8]
repo = "ByteDance/AnimateDiff-Lightning"
ckpt = f"animatediff_lightning_{step}step_diffusers.safetensors"
base = "emilianJR/epiCRealism"  # Choose to your favorite base model.








# Title and description for your Streamlit app
st.title('Web Page Summarizer and Image Generator')
st.markdown("""
            This app scrapes a webpage, summarizes its content, and generates image shorts.
            Enter the URL of the webpage you want to process.
            """)

# Input field for URL
url = st.text_input('Enter URL of the webpage')

# Button to trigger processing
if st.button('Process'):
    if url:
        try:
            st.write("Starting to scrap the webpage...")
            scraped_content = scrap(url)  # Assuming scrap function is implemented
            st.write("Webpage scraped")
            st.write(scraped_content)

            # Initialize T5 model and tokenizer
            st.write("Loading model...")
            model = T5ForConditionalGeneration.from_pretrained('t5-large')
            st.write("Model loaded")

            st.write("Loading tokenizer...")
            tokenizer = T5Tokenizer.from_pretrained('t5-large')
            st.write("Tokenizer loaded")

            st.write("Summarizing the content...")
            summary = summarize(scraped_content, model, tokenizer)
            st.write("Content summarized")
            st.write(summary)

            st.write('Validating the summary by Bart model...')
            summary_bart = generate_summary_bart(scraped_content,bart_tokenizer,bart_model)
            # cosine_result=cosine_similarity(summary,summary_bart)
            st.write(summary_bart)

            # Save the summary to a text file
            summary_file_path = 'summary.txt'
            with open(summary_file_path, 'w') as file:
                file.write(summary_bart)
            st.write(f"Summary saved to {summary_file_path}")
            st.stop()
        
        except Exception as e:
            st.error(f'Error occurred: {str(e)}')
            st.write(f'Error details: {str(e)}')

else:
    st.write("Waiting for user input")
