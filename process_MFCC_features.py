from collections import defaultdict
import matplotlib.pyplot as plt
import librosa
import time
import os
import io
import warnings
import multiprocessing
from PIL import Image
from sklearn.preprocessing import scale

def info(title):
    print(title)
    print('module name: ', __name__)
    print('parent process: ', os.getppid())
    print('process id: ', os.getpid())

def fig2img(fig):
    buf = io.BytesIO()
    try:
        fig.savefig(buf)
        buf.seek(0)
        img = Image.open(buf)
        return img
    except Exception as e:
        print(f"Error converting Matplotlib figure to image: {e}")
        return None

def WavtoMFCCs(file_path, sample_rate, n_fft,
               hop_length, n_mfcc):

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
    try:
        source_path = os.path.join(base_path, split_path)
        wav_files = [f for f in os.listdir(source_path) if os.path.isfile(os.path.join(source_path, f))]
        return wav_files
    except Exception as e:
        print(f"Failed to find file(s) for {source_path}")
        return None


def process_batch(batch_files, base_path, split_path, base_save_path,
                  sample_rate, n_fft, hop_length, n_mfcc):
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
                resized_image = img.resize((width, height))

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