"""
Audio Feature Extraction Pipeline

This script processes audio files to extract Mel-frequency cepstral coefficients (MFCCs) 
and converts them into scaled images.

The script uses multiprocessing to parallelize the feature extraction process,
improving performance on multi-core systems.

Main components:
- Audio feature extraction using librosa
- Visualization of MFCCs using matplotlib
- Multiprocessing for parallel execution
- Image conversion and saving

Dependencies:
- collections
- matplotlib
- librosa
- time
- os
- io
- warnings
- multiprocessing
- PIL
- sklearn
"""

from collections import defaultdict
import matplotlib.pyplot as plt
import librosa
import time
import os
import io
import warnings
import multiprocessing
from PIL import Image, ImageChops
from sklearn.preprocessing import scale

def info(title):
    """
    Prints process information for debugging purposes.
    
    Args:
        title (str): Title to identify the process information section
    """
    print(title)
    print('module name: ', __name__)
    print('parent process: ', os.getppid())
    print('process id: ', os.getpid())

def fig2img(fig):
    """
    Converts a matplotlib figure to a PIL Image object.
    
    Args:
        fig (matplotlib.figure.Figure): The matplotlib figure to convert
    
    Returns:
        PIL.Image or None: The converted image, or None if conversion fails
    """
    buf = io.BytesIO()
    try:
        fig.savefig(buf)
        buf.seek(0)
        img = Image.open(buf)
        return img
    except Exception as e:
        print(f"Error converting Matplotlib figure to image: {e}")
        return None
    
def trim(im):
    """
    Removes background/whitespace from an image by detecting the content boundaries.
    
    This function attempts to detect the actual content in an image by creating a 
    background image of the same color as the corner pixel, finding the difference
    between the original image and this background, and then cropping to the 
    bounding box of the differences. If the process fails in the current color mode,
    it recursively attempts the process in RGB mode.

    shoutout: https://stackoverflow.com/questions/10615901/trim-whitespace-using-pil 
    
    Args:
        im (PIL.Image): The input image to trim
        
    Returns:
        PIL.Image: The trimmed image with background/whitespace removed
        
    Raises:
        Various PIL exceptions could be caught and logged within the function
    
    """
    try:
        bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
        diff = ImageChops.difference(im, bg)
        diff = ImageChops.add(diff, diff, 2.0, -100)
        bbox = diff.getbbox()
        if bbox:
            return im.crop(bbox)
        else: 
            # Failed to find the borders, convert to "RGB"        
            return trim(im.convert('RGB'))
    except Exception as e:
        print(f"Error triming whitespace on image: {e}")

def WavtoMFCCs(file_path, sample_rate, n_fft,
               hop_length, n_mfcc):
    """
    Extracts Mel-frequency cepstral coefficients (MFCCs) from an audio file.
    
    Args:
        file_path (str): Path to the audio file
        sample_rate (int): Target sample rate for audio loading
        n_fft (int): FFT window size for MFCC extraction
        hop_length (int): Number of samples between successive frames
        n_mfcc (int): Number of MFCCs to extract
    
    Returns:
        numpy.ndarray or None: Scaled MFCCs if successful, None otherwise
    """
    try:
        y, sr = librosa.load(file_path, sr=sample_rate)
    
        mfcc = librosa.feature.mfcc(y=y, sr=sr,
                                n_fft=n_fft, n_mfcc=n_mfcc,
                                hop_length=hop_length)
        mfccs = scale(mfcc, axis=1)
        return mfccs
    except librosa.LibrosaError as e:
        print(f"LibrosaError processing {file_path}: {e}")
        return None
    except FileNotFoundError:
        print(f"FileNotFoundError: Audio file not found at {file_path}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while processing {file_path}: {e}")
        return None

def getAudioFiles(base_path, split_path):
    """
    Gets a list of audio files from a specified directory.
    
    Args:
        base_path (str): The base directory path
        split_path (str): The subdirectory containing audio files
    
    Returns:
        list or None: List of audio filenames if successful, None otherwise
    """
    try:
        source_path = os.path.join(base_path, split_path)
        wav_files = [f for f in os.listdir(source_path) if os.path.isfile(os.path.join(source_path, f))]
        return wav_files
    except Exception as e:
        print(f"Failed to find file(s) for {source_path}")
        return None


def process_batch(batch_files, base_path, split_path, base_save_path,
                  sample_rate, n_fft, hop_length, n_mfcc):
    """
    Processes a batch of audio files, extracting MFCCs and saving as images.
    
    This function is designed to be run by a worker process in the multiprocessing pool.
    
    Args:
        batch_files (list): List of audio filenames to process
        base_path (str): Base directory containing audio files
        split_path (str): Subdirectory for the specific dataset split
        base_save_path (str): Base directory to save output images
        sample_rate (int): Target sample rate for audio processing
        n_fft (int): FFT window size
        hop_length (int): Number of samples between successive frames
        n_mfcc (int): Number of MFCCs to extract
    """
    info(f"processing batch for {split_path}")
    warnings.filterwarnings("ignore") # mute mfcc scaling warning, scale is not exact but close enough
    batch_size = len(batch_files)
    for i, audio_file in enumerate(batch_files):
        file_path = os.path.join(base_path, split_path, audio_file)
        file_title = audio_file.split('.')[0]
        mfccs = WavtoMFCCs(file_path,
                sample_rate, n_fft, hop_length, n_mfcc)                      
        if mfccs is None:
            print(f"Failed to extract features: {file_title}\nMoving onto next file")
            continue

        try:
            librosa.display.specshow(mfccs, sr=sample_rate,
                                    n_fft=n_fft,
                                    hop_length=hop_length)
            fig = plt.gcf()
            img = fig2img(fig)

            if img:
                # Size the image to 200x200 for consistent data size
                width = 200
                height = 200
                img = trim(img)
                resized_image = img.resize((width, height), resample=Image.LANCZOS)

                save_to_path = os.path.join(base_save_path, split_path)
                os.makedirs(save_to_path, exist_ok=True)
                save_to_path = os.path.join(save_to_path, file_title)
                resized_image.save(save_to_path + '.png')
                print(f"Process {os.getpid()} saved MFCC for {file_title} ({i+1}/{batch_size}) in {split_path}")
            else:
                print(f"Process {os.getpid()} saved MFCC for {file_title} ({i+1}/{batch_size}) in {split_path}")

        except Exception as e:
            print(f"Process {os.getpid()} failed to save image for {file_title}: {e}")

def process_split(base_path, split_path, base_save_path, num_processes):
    """
    Processes an entire dataset split using multiple processes.
    
    Distributes the workload of processing audio files across multiple CPU cores.
    
    Args:
        base_path (str): Base directory containing audio files
        split_path (str): Subdirectory for the specific dataset split
        base_save_path (str): Base directory to save output images
        num_processes (int): Number of parallel processes to use
    """
    info(f'preparing to process split: {split_path}')
    sample_rate = 16000
    frame_len = 0.025 # 25ms
    hop_len = 0.01 # 10ms
    n_fft = int(sample_rate * frame_len)
    hop_length = int(sample_rate * hop_len)
    n_mfcc = 13

    wav_files = getAudioFiles(base_path, split_path)
    if wav_files is None:
        print(f"No WAV files found in {os.path.join(base_path, split_path)}")
        return
    
    total_files = len(wav_files)
    chunk_size = (total_files + num_processes-1) // num_processes
    file_chunks = [wav_files[i * chunk_size:(i + 1) * chunk_size] for i in range(num_processes)]
    pool_args = [(chunk, base_path, split_path, base_save_path, sample_rate, n_fft, hop_length, n_mfcc)
                 for chunk in file_chunks if chunk]
    
    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.starmap(process_batch, pool_args)

    print(f"Finished processing split: {split_path}")


if __name__ == '__main__':
    """
    Main execution block.
    
    Sets up and executes parallel processing of audio files from multiple dataset splits.
    Each dataset split is processed in its own process, and within each process,
    multiple CPU cores are utilized through a process pool.
    """
    # Testing single cpu processing speed
    start_time = time.time()
    base_path = 'Datasets_Wavs/'
    base_save_path = 'Datasets_Image/MFCC/'
    source_dirs = ['ASVspoof2017_V2_train', 'ASVspoof2017_V2_eval', 'ASVspoof2017_V2_dev']

    # The arguments for each process request
    # Sample single arg tuple: ('Datasets_Wavs/', 'ASVspoof2017_V2_train', 'Datasets_Image/MFCC/')
    args = [] 
    for split in source_dirs:
        args.append((base_path, split, base_save_path))

    num_cores = multiprocessing.cpu_count()
    print(f"Using {num_cores} CPU cores.")
    processes = []
    for split in source_dirs:
        process = multiprocessing.Process(target=process_split,
                                          args=(base_path, split, base_save_path, num_cores))
        processes.append(process)
        process.start()
    
    for process in processes:
        process.join()

    end_time = time.time()
    execution_time = end_time - start_time

    # Total runtime for 1000 sample no concurrency: about 9 hrs
    print(f"Total execution time processed in parallel: {execution_time:.4f}")