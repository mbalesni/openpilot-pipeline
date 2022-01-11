from train.dataloader import load_transformed_video
import sys
from utils import printf
import time


if __name__ == "__main__":

    # mock model
    simulated_forward_time = 0.5

    printf("Checking loader shapes...")
    printf()
    for epoch in range(5):
        time_start = time.time()
        input_frames, rgb_frames = load_transformed_video('/home/nikita/data/2021-09-14--09-19-21/25/')
        # NOTE: no need to use `create_img_plot_canvas` function anymore — just feed the `rgb_frames` one-by-one to the `draw_path`.

        printf(f'{time.time() - time_start:.2f}s – to load batch.')
        printf(f'Input frames: {input_frames.shape}. RGB frames (transformed & downsampled): {rgb_frames.shape}')

        # <model goes here>
        time.sleep(simulated_forward_time)
        time_start = time.time()
