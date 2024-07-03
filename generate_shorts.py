from pathlib import Path
from PIL import Image
import re
import imageio
from diffusers.utils import export_to_gif



# Function to split the summary text into individual sentences
def split_into_sentences(summary):
    # Using regular expressions to split text by sentence-ending punctuation
    sentences = re.split(r'(?<=[.!?]) +', summary.strip())
    return [sentence for sentence in sentences if sentence]


def generate_shorts(summary,pipe,step):
    # pipe = pipe.to("cuda")  # Use GPU if available

    # Split the summary text into sentences
    prompts = split_into_sentences(summary)
    print('splitted into sentences')

    # Ensure both directories exist
    frames_output_dir = Path('Generated-Frames')
    frames_output_dir.mkdir(parents=True, exist_ok=True)

    videos_output_dir = Path('Generated-Shorts')
    videos_output_dir.mkdir(parents=True, exist_ok=True)

    output_gifs=[]
    for i, prompt in enumerate(prompts):
        output = pipe(prompt=f'A news from Nepal {prompt}', guidance_scale=1.0, num_inference_steps=step)
        gif_path = frames_output_dir / f"animation{i}.gif"
        output_gifs.append(export_to_gif(output.frames[0], str(gif_path)))

    # Merge GIFs into a single file
    merged_output_path = videos_output_dir / "video.gif"
    with imageio.get_writer(merged_output_path, mode='I',loop=0) as writer:
        for gif_file in output_gifs:
            gif = imageio.get_reader(gif_file)
            for frame in gif:
                writer.append_data(frame)
    
    return merged_output_path








    
