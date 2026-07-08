"""
video_utils.py

Utility functions for ultrasound videos.
"""

import cv2
import numpy as np


def read_video(video_path):
    """
    Read all frames from a video.
    """

    cap = cv2.VideoCapture(video_path)

    frames = []

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (224, 224))

        frames.append(frame)

    cap.release()

    return frames


def sample_frames(frames, num_frames=16):
    """
    Uniformly sample frames from a video.
    """

    total_frames = len(frames)

    if total_frames == 0:

        return np.zeros(
            (num_frames, 224, 224, 3),
            dtype=np.uint8,
        )

    if total_frames < num_frames:

        while len(frames) < num_frames:
            frames.append(frames[-1])

        return np.array(frames)

    indices = np.linspace(
        0,
        total_frames - 1,
        num_frames,
        dtype=int,
    )

    sampled_frames = [frames[i] for i in indices]

    return np.array(sampled_frames)