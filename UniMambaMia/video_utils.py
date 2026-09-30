# UniMambaMia-AV
# Copyright (c) 2026-present NAVER Cloud Corp.
# MIT license
#
# Modified from https://github.com/naver-ai/mambamia/blob/main/MambaMia/video_utils.py

import os
import struct

os.environ['DECORD_EOF_RETRY_MAX'] = "20480"

import math
from decord import VideoReader, AudioReader, cpu
import numpy as np
from PIL import Image
import gc
import ctypes

# glibc malloc_trim: force worker processes to return freed memory to the OS.
try:
    _libc = ctypes.CDLL("libc.so.6")

    def malloc_trim():
        _libc.malloc_trim(0)
except Exception:

    def malloc_trim():
        pass


def _drop_file_cache(filepath):
    """Ask the kernel to drop the page cache of ``filepath`` (posix_fadvise DONTNEED).

    decord keeps the decoded video/audio file in the page cache; inside a cgroup the page
    cache counts against the memory limit and can trigger OOM kills in long runs.
    """
    try:
        fd = os.open(filepath, os.O_RDONLY)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fd)
    except Exception:
        pass


def _read_audio_isolated(video_file_path, sample_rate, video_length):
    """Read the whole soundtrack with decord's AudioReader inside a forked child process.

    decord 0.6.0's AudioReader leaks memory at the C++ level (see decord issues #153, #189,
    #279); ``del ar`` does not release the internal FFmpeg buffers, so DataLoader workers grow
    by several MB per sample. Running the reader in a short-lived fork and returning the
    waveform through a pipe lets the OS reclaim everything when the child exits.

    Returns:
        (waveform float32 [N], duration seconds). Falls back to silence on failure.
    """
    pipe_r, pipe_w = os.pipe()
    pid = os.fork()

    if pid == 0:
        # -- child --
        os.close(pipe_r)
        try:
            ar = AudioReader(video_file_path, ctx=cpu(0), sample_rate=sample_rate, mono=True)
            full_audio = ar[:].asnumpy().flatten()
            audio_duration = float(ar.duration())
            del ar
            # Protocol: [8 bytes float64 duration] + [N*4 bytes float32 audio]
            with os.fdopen(pipe_w, 'wb') as f:
                f.write(struct.pack('d', audio_duration))
                f.write(full_audio.astype(np.float32).tobytes())
        except Exception:
            with os.fdopen(pipe_w, 'wb') as f:
                f.write(struct.pack('d', 0.0))
        os._exit(0)

    # -- parent --
    os.close(pipe_w)
    with os.fdopen(pipe_r, 'rb') as f:
        data = f.read()
    os.waitpid(pid, 0)
    _drop_file_cache(video_file_path)

    if len(data) >= 8:
        audio_duration = struct.unpack('d', data[:8])[0]
        if audio_duration > 0 and len(data) > 8:
            full_audio = np.frombuffer(data[8:], dtype=np.float32).copy()
            return full_audio, audio_duration

    # fallback: silence of the video length
    return np.zeros(int(sample_rate * video_length + 1), dtype=np.float32), video_length


class VideoReaderWrapper(VideoReader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seek(0)

    def __getitem__(self, key):
        frames = super().__getitem__(key)
        self.seek(0)
        return frames


def sample_video_audio(
    FPS,
    video_file_path,
    MAX_Frame,
    cpu_idx=1,
    use_audio=False,
    sample_rate=16000,
    debug=False,
    twice_for_long=None,
    fourtime_for_long=None,
    eighttime_for_long=None,
    max_video_duration=None,
):
    """
    Samples frames from a video at the specified FPS with a maximum frame limit and optionally
    returns the soundtrack as a list of 30-second waveform chunks.

    Args:
        FPS (float): Desired frames per second to sample from the video.
        video_file_path (str): Path to the video file.
        MAX_Frame (int): Maximum number of frames to sample.
        cpu_idx (int, optional): CPU index to use for decoding. Defaults to 1.
        use_audio (bool, optional): Whether to read the audio track. Defaults to False.
        sample_rate (int, optional): Sampling rate for audio. Defaults to 16000.
        debug (bool, optional): If True, prints debug information. Defaults to False.
        max_video_duration (float, optional): Skip (raise) videos longer than this many seconds.
            Useful to bound audio memory during training. None = no limit.

    Returns:
        tuple:
            - list_of_frames (np.ndarray): Sampled frames, shape (num_frames, H, W, 3), RGB uint8.
            - list_of_audio_chunks (List[np.ndarray]): 30 s waveform chunks (16 kHz, mono, float32),
              matching the 30 s windows of the Whisper feature extractor. Empty if use_audio=False.
            - actual_FPS (float): The actual FPS used for sampling.
            - video_length (float): Total length of the video in seconds.
    """
    if debug:
        print(f"[DEBUG] Checking if video file exists at: {video_file_path}")
    if not os.path.isfile(video_file_path):
        raise FileNotFoundError(f"Video file not found: {video_file_path}")

    if debug:
        print(f"[DEBUG] Initializing CPU context with CPU index: {cpu_idx}")
    ctx = cpu(cpu_idx)

    if debug:
        print("[DEBUG] Initializing VideoReader.")
    try:
        vr = VideoReaderWrapper(video_file_path, ctx=ctx, num_threads=1)
    except Exception as e:
        raise IOError(f"Error reading video file: {e}")

    # Get video properties
    video_fps = vr.get_avg_fps()
    total_frames = len(vr)
    # Calculate accurate video length from last frame's timestamp
    last_frame_end_time = vr.get_frame_timestamp(-1)[1]
    video_length = last_frame_end_time  # In seconds

    # Skip over-long videos (audio memory guard)
    if max_video_duration is not None and video_length > max_video_duration:
        vr = None
        del vr
        gc.collect()
        raise ValueError(f"Video too long ({video_length:.1f}s > {max_video_duration:.1f}s), skipping: {video_file_path}")

    if twice_for_long is not None:
        if video_length > twice_for_long:
            MAX_Frame *= 2
    if fourtime_for_long is not None:
        if video_length > fourtime_for_long:
            MAX_Frame *= 4
    if eighttime_for_long is not None:
        if video_length > eighttime_for_long:
            MAX_Frame *= 8

    if debug:
        print(f"[DEBUG] Video FPS: {video_fps}")
        print(f"[DEBUG] Total frames: {total_frames}")
        print(f"[DEBUG] Video length: {video_length:.2f} seconds")

    full_audio = None
    if use_audio:
        if debug:
            print("[DEBUG] Reading audio via isolated subprocess.")
        full_audio, _audio_dur = _read_audio_isolated(video_file_path, sample_rate, video_length)
        if debug:
            print(f"[DEBUG] Audio read: {len(full_audio)} samples, duration={_audio_dur:.2f}s")
            if abs(_audio_dur - video_length) > 1.0:
                print(f"[WARNING] Audio duration ({_audio_dur:.2f}s) and video duration ({video_length:.2f}s) differ by more than 1.0 seconds.")

    # Calculate desired number of frames
    desired_num_frames = math.ceil(video_length * FPS)

    # Determine actual FPS based on MAX_Frame
    if desired_num_frames > MAX_Frame:
        actual_FPS = MAX_Frame / video_length
        actual_num_frames = MAX_Frame
        if debug:
            print(f"[DEBUG] Desired frames ({desired_num_frames}) exceed MAX_Frame ({MAX_Frame}).")
            print(f"[DEBUG] Adjusting FPS from {FPS} to {actual_FPS:.2f} to respect MAX_Frame.")
    else:
        actual_FPS = FPS
        actual_num_frames = desired_num_frames
        if debug:
            print(f"[DEBUG] Desired frames ({desired_num_frames}) within MAX_Frame ({MAX_Frame}). Using FPS={FPS}.")

    # Generate timestamps for each desired frame based on actual_FPS
    sampled_times = np.linspace(0, video_length, num=actual_num_frames, endpoint=False)
    if debug:
        print(f"[DEBUG] Sampled frame times: {sampled_times}")

    # Function to find the closest frame index for a given timestamp
    def find_closest_frame(time_stamp):
        frame_idx = min(int(np.round(time_stamp * video_fps)), total_frames - 1)
        return frame_idx

    frame_indices = [find_closest_frame(t) for t in sampled_times]
    if debug:
        print(f"[DEBUG] Frame indices to sample: {frame_indices}")

    list_of_frames = vr.get_batch(frame_indices).asnumpy()  # (num_frames, H, W, 3)
    vr.seek(0)

    # Release the VideoReader immediately
    vr = None
    del vr
    _drop_file_cache(video_file_path)

    if debug:
        print(f"[DEBUG] Extracted {len(list_of_frames)} frames.")

    # Audio: split into 30 s chunks (one Whisper window each)
    list_of_audio_chunks = []
    if use_audio and full_audio is not None:
        chunk_size = sample_rate * 30
        total_samples = len(full_audio)
        for start in range(0, total_samples, chunk_size):
            end = min(start + chunk_size, total_samples)
            list_of_audio_chunks.append(full_audio[start:end].copy())
        del full_audio

        if debug:
            print(f"[DEBUG] Total audio chunks: {len(list_of_audio_chunks)}")

    gc.collect()
    malloc_trim()
    _drop_file_cache(video_file_path)

    return list_of_frames, list_of_audio_chunks, actual_FPS, video_length
