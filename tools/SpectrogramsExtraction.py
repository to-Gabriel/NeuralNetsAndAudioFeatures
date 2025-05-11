import os
import wave
import numpy as np
import matplotlib.pyplot as plt
import traceback

# --- Spectrogram Function (Mostly unchanged, added more robustness) ---
def analyze_and_save_spectrogram_image(
    file_path,
    output_image_path,
    image_dim=(200, 200),
    nfft=1024,
    noverlap=512
):
    """
    Analyzes a WAV file, generates its spectrogram, and saves it as a PNG image.

    Args:
        file_path (str): Path to the input WAV file.
        output_image_path (str): Path where the output PNG image will be saved.
        image_dim (tuple): Desired dimensions (width, height) of the output image in pixels.
        nfft (int): The number of data points used in each block for the FFT.
                    Higher values give more frequency detail, lower values more time detail.
        noverlap (int): The number of points of overlap between blocks.
                         Should be less than nfft.
    """
    print(f"Processing: {os.path.basename(file_path)}")
    try:
        # Check if file exists before trying to open
        if not os.path.exists(file_path):
            print(f"Error: Input file not found: {file_path}")
            return

        with wave.open(file_path, 'rb') as wf:
            sample_width = wf.getsampwidth()
            frame_rate = wf.getframerate()
            num_frames = wf.getnframes()

            # Basic validation of wave file properties
            if sample_width != 2:
                print(f"Warning: Skipping {file_path}. Expected 16-bit (2 bytes) audio, found {sample_width} bytes.")
                return
            if frame_rate <= 0:
                print(f"Warning: Skipping {file_path}. Invalid frame rate: {frame_rate}")
                return
            if num_frames <= 0:
                 print(f"Warning: Skipping {file_path}. No frames found.")
                 return

            raw = wf.readframes(num_frames)
            if not raw: # Check if readframes returned any data
                 print(f"Warning: Skipping {file_path}. Failed to read frames.")
                 return

        data = np.frombuffer(raw, dtype=np.int16)
        if data.size == 0:
            print(f"Warning: Skipping {file_path}. No data after buffer conversion.")
            return

        # Check if data is essentially silent or constant, which might cause specgram issues
        if np.all(data == data[0]):
             print(f"Warning: Skipping {file_path}. Audio data appears to be constant (silence).")
             return

        # --- Spectrogram Generation ---
        target_w_pixels, target_h_pixels = image_dim
        dpi = 100 # Using a fixed DPI for consistent figure sizing

        # Create figure with precise dimensions in inches based on target pixels and DPI
        fig = plt.figure(figsize=(target_w_pixels / dpi, target_h_pixels / dpi), dpi=dpi)
        # Add axes that fills the entire figure area
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off') # Turn off axis lines and labels

        try:
            # Generate the spectrogram
            # scale='dB' is often preferred for audio visualization
            Pxx, freqs, bins, im = ax.specgram(data, NFFT=nfft, Fs=frame_rate, noverlap=noverlap, scale='dB', cmap='viridis')
        except ValueError as ve:
             print(f"Error during specgram generation for {file_path}: {ve}. Skipping.")
             plt.close(fig) # Close the figure even if specgram fails
             return


        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_image_path), exist_ok=True)

        # Save the figure without extra whitespace
        plt.savefig(output_image_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
        plt.close(fig) # Close the figure explicitly to free memory
        print(f"Saved: {output_image_path}")

    except wave.Error as e:
         print(f"Error opening or reading wave file {file_path}: {e}")
    except FileNotFoundError: # Catching this explicitly if os.path.exists fails (race condition)
         print(f"Error: Input file disappeared before processing: {file_path}")
    except Exception as e:
        print(f"An unexpected error occurred processing {file_path}: {e}")
        traceback.print_exc() # Print detailed traceback for debugging
    finally:
        # Ensure all matplotlib figures are closed in case of unhandled exceptions
         plt.close('all')


# --- Main Processing Logic based on Protocol Files ---

# Define base paths accurately based on the prompt
protocol_base_dir = '/home/madan005/dev/deeplearning/ASVspoof2017_V2/protocol_V2'
audio_base_dir = '/data/users/madan005/ASVspoof2017_V2/dataset'
output_root = './spectrogram_images_asv2017' # Specify the root directory for output images
image_dim = (200, 200) # Desired output image dimensions

# Define the protocol files and their corresponding data splits and audio subfolders
protocol_mapping = {
    'train': {'protocol': 'train.trn.txt', 'audio_folder': 'train'},
    'dev':   {'protocol': 'dev.trl.txt',   'audio_folder': 'dev'},
    'eval':  {'protocol': 'eval.trl.txt',  'audio_folder': 'eval'}
}

# Process each data split defined in the mapping
for split_name, split_info in protocol_mapping.items():
    protocol_file = os.path.join(protocol_base_dir, split_info['protocol'])
    audio_split_folder = os.path.join(audio_base_dir, split_info['audio_folder'])

    print(f"\n--- Processing split: {split_name} ---")
    print(f"Protocol file: {protocol_file}")
    print(f"Audio folder: {audio_split_folder}")

    if not os.path.exists(protocol_file):
        print(f"Warning: Protocol file not found: {protocol_file}. Skipping this split.")
        continue

    if not os.path.isdir(audio_split_folder):
         print(f"Warning: Audio directory not found: {audio_split_folder}. Skipping this split.")
         continue


    try:
        with open(protocol_file, 'r') as f_protocol:
            line_count = 0
            processed_count = 0
            for line in f_protocol:
                line_count += 1
                line = line.strip()
                if not line: # Skip empty lines
                    continue

                try:
                    # Split the line based on whitespace
                    parts = line.split()
                    # Expecting at least 2 columns: File ID and Speech Type
                    if len(parts) < 2:
                        print(f"Warning: Line {line_count}: Malformed line - expected at least 2 columns, got {len(parts)}. Content: '{line}'. Skipping.")
                        continue

                    file_id = parts[0]       # e.g., T_1000001.wav (contains extension)
                    speech_type = parts[1]   # 'genuine' or 'spoof'

                    # Validate speech type
                    if speech_type not in ['genuine', 'spoof']:
                        print(f"Warning: Line {line_count}: Unknown speech type '{speech_type}' for file ID {file_id}. Skipping.")
                        continue

                    # --- CORRECTION 1 ---
                    # Construct the full path to the audio file
                    # Use file_id directly as it includes the .wav extension
                    audio_filename = file_id
                    audio_file_path = os.path.join(audio_split_folder, audio_filename)

                    # Construct the output directory structure: <output_root>/<split>/<speech_type>/
                    output_dir = os.path.join(output_root, split_name, speech_type)

                    # --- CORRECTION 2 ---
                    # Construct the output image filename: <file_id_without_wav>.png
                    base_output_filename = os.path.splitext(file_id)[0] # Get filename part before extension
                    output_image_filename = f"{base_output_filename}.png" # Add .png to base name
                    output_image_path = os.path.join(output_dir, output_image_filename)

                    # Generate and save the spectrogram image
                    analyze_and_save_spectrogram_image(
                        audio_file_path,
                        output_image_path,
                        image_dim=image_dim
                        # nfft, noverlap will use the defaults defined in the function
                    )
                    processed_count +=1

                except Exception as line_e:
                    print(f"Error processing line {line_count} ('{line}') from {protocol_file}: {line_e}")
                    # Optionally print traceback for debugging line-specific errors
                    # traceback.print_exc()

            print(f"--- Finished split: {split_name}. Processed {processed_count}/{line_count} lines from protocol file. ---")

    except IOError as e:
        print(f"Error reading protocol file {protocol_file}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while processing the {split_name} split: {e}")
        traceback.print_exc()


print("\nAll splits processed. Spectrogram generation finished.")