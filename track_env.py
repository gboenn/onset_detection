import numpy as np
import soundfile as sf
from scipy.io import wavfile
import sys
import os
import matplotlib.pyplot as plt
from matplotlib import ticker


def track_amplitude(input_file, output_file, normalize=False):
    # Stage 1: Read WAV file
    sample_rate, audio_data = wavfile.read(input_file)
    
    # Stage 2: Handle channel configuration
    if len(audio_data.shape) == 1:
        channels = 1
        audio_data = audio_data.reshape(-1, 1)
    else:
        channels = audio_data.shape[1]
    
    # Stage 3: Generate temporal axis
    n_samples = audio_data.shape[0]
    time_vector = np.arange(n_samples) / sample_rate
    
    # Stage 4: Optional amplitude normalization
    if normalize:
        if np.issubdtype(audio_data.dtype, np.integer):
            max_val = np.iinfo(audio_data.dtype).max
            audio_data = audio_data.astype(float) / max_val
    
    # Stage 5: Data formatting and export
    output_matrix = np.column_stack((time_vector.reshape(-1,1), audio_data))
    
    header = "Time(s)" + "\tChannel" + "\tChannel".join(map(str, range(1, channels+1)))
    fmt = ["%.6f"] + ["%.6f" if normalize else "%d"] * channels
    
    np.savetxt(output_file, output_matrix, 
              delimiter='\t', 
              header=header,
              fmt=fmt,
              comments='')

# import numpy as np

def track_rms(input_file, output_file, block_size=1024, normalize=False):
    # Read audio file with original sampling
    audio_data, sample_rate = sf.read(input_file, always_2d=True)
        
    # Handle mono/stereo conversion
    channels = audio_data.shape[1] if len(audio_data.shape) > 1 else 1
    audio_data = audio_data.reshape(-1, channels)
    
    # Normalization protocol
    if normalize and np.issubdtype(audio_data.dtype, np.integer):
        max_val = np.iinfo(audio_data.dtype).max
        audio_data = audio_data.astype(np.float32) / max_val

    # Initialize output containers
    times = []
    rms_values = []
    
    # Block processing loop
    for i in range(0, len(audio_data), block_size):
        block = audio_data[i:i+block_size]
        frame_time = i / sample_rate
        
        # Vectorized RMS calculation
        squared = np.square(block)
        mean_squared = np.mean(squared, axis=0)
        rms = np.sqrt(mean_squared)
        
        times.append(frame_time)
        rms_values.append(rms)

    # Format output matrix
    output_matrix = np.column_stack((np.array(times), np.array(rms_values)))
    
    # Generate dynamic headers
    headers = ["Time(s)"] + [f"RMS_Ch{n+1}" for n in range(channels)]
    
    # Save with precision formatting
    np.savetxt(output_file, output_matrix,
              delimiter='\t',
              header='\t'.join(headers),
              fmt='%.6f',
              comments='')

def track_rms_with_plot(input_file, output_file, block_size=1024, normalize=False):
    """Analyze audio RMS in blocks and generate time-domain plot"""
    
    # Read audio with correct parameter order
    audio_data, sample_rate = sf.read(input_file, always_2d=True)
    
    # Ensure 2D array for consistent processing
    channels = audio_data.shape[1] if len(audio_data.shape) > 1 else 1
    audio_data = audio_data.reshape(-1, channels)
    
    # Normalization handling
    if normalize and np.issubdtype(audio_data.dtype, np.integer):
        max_val = np.iinfo(audio_data.dtype).max
        audio_data = audio_data.astype(np.float32) / max_val

    # Initialize containers
    times = []
    rms_values = []
    
    # Block processing
    for i in range(0, len(audio_data), block_size):
        block = audio_data[i:i+block_size]
        frame_time = i / sample_rate
        
        # Vectorized RMS calculation
        rms = np.sqrt(np.mean(np.square(block), axis=0))
        
        times.append(frame_time)
        rms_values.append(rms)

    # Convert to arrays
    times_array = np.array(times)
    rms_array = np.array(rms_values)

    # Create plot
    plt.figure(figsize=(12, 6))
    
    if channels == 1:
        plt.plot(times_array, rms_array, label='RMS Amplitude')
    else:
        for ch in range(channels):
            plt.plot(times_array, rms_array[:, ch], 
                    label=f'Channel {ch+1} RMS')

    plt.title(f'Time vs RMS Amplitude ({block_size} Sample Blocks)')
    plt.xlabel('Time (seconds)')
    plt.ylabel('RMS Amplitude' + (' (Normalized)' if normalize else ''))
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    # Save plot and data
    plt.savefig('rms_plot.png', dpi=300, bbox_inches='tight')
    np.savetxt(output_file, np.column_stack((times_array, rms_array)),
              delimiter='\t',
              header='Time(s)\t' + '\t'.join([f'RMS_Ch{n+1}' for n in range(channels)]),
              fmt='%.6f',
              comments='')

    plt.show()
    plt.close()

# Usage example
# track_rms_with_plot("input.wav", "rms_values.txt", normalize=True)

def track_rms_with_metadata(input_file, output_file, block_size=1024, normalize=False):
    """Analyze audio RMS with full file metadata visualization"""
    
    # Read audio with metadata extraction
    audio_data, sample_rate = sf.read(input_file, always_2d=True)
    file_info = sf.info(input_file)
    
    # Extract technical specifications
    subtype = file_info.subtype
    duration = file_info.duration
    channels = file_info.channels
    frames = file_info.frames
    
    # Determine bit depth from subtype
    if 'PCM' in subtype:
        bit_depth = f"{subtype.split('_')[-1]}-bit PCM"
    elif subtype == 'FLOAT':
        bit_depth = "32-bit Float"
    elif subtype == 'DOUBLE':
        bit_depth = "64-bit Double"
    else:
        bit_depth = subtype.replace('_', ' ').title()

    # Signal conditioning
    audio_data = audio_data.reshape(-1, channels)
    if normalize and np.issubdtype(audio_data.dtype, np.integer):
        max_val = np.iinfo(audio_data.dtype).max
        audio_data = audio_data.astype(np.float32) / max_val

    # RMS calculation
    times = []
    rms_values = []
    for i in range(0, len(audio_data), block_size):
        block = audio_data[i:i+block_size]
        times.append(i / sample_rate)
        rms_values.append(np.sqrt(np.mean(np.square(block), axis=0)))

    # Create figure with technical annotations
    plt.figure(figsize=(13, 7))
    ax = plt.gca()
    
    # Formatting for technical audience
    plt.plot(times, rms_values, linewidth=1.0)
    plt.title("Time vs RMS Amplitude Characteristics", pad=20, fontsize=14)
    
    # Create metadata box
    meta_text = (
        f"File: {os.path.basename(input_file)}\n"
        f"Sample Rate: {sample_rate/1000:.1f} kHz\n"
        f"Bit Depth: {bit_depth}\n"
        f"Duration: {duration:.2f}s\n"
        f"Channels: {channels}\n"
        f"Analysis Block: {block_size} samples\n"
        f"Total Frames: {frames:,}"
    )
    
    plt.text(0.98, 0.95, meta_text,
             transform=ax.transAxes,
             horizontalalignment='right',
             verticalalignment='top',
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='#CCCCCC'),
             fontfamily='monospace',
             fontsize=9)

    # Configure axes
    ax.set_xlabel('Time (seconds)', fontsize=10)
    ax.set_ylabel('RMS Amplitude' + (' (Normalized)' if normalize else ''), fontsize=10)
    ax.xaxis.set_major_formatter(ticker.EngFormatter(unit='s'))
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0) if normalize else None)
    ax.grid(True, which='both', linestyle='--', alpha=0.6)
    
    # Save outputs
    plt.savefig('rms_analysis.png', dpi=300, bbox_inches='tight')
    np.savetxt(output_file, np.column_stack((times, rms_values)),
              delimiter='\t',
              header='Time(s)\t' + '\t'.join([f'RMS_Ch{n+1}' for n in range(channels)]),
              fmt='%.6f',
              comments='')
    plt.close()

# Usage example
# track_rms_with_metadata("input.wav", "rms_analysis.txt")

def main ():
    if (len(sys.argv) < 3):
        print("Usage: python[3.7] track_env.py <input wav> <output text>")
        return

    path = sys.argv[1]
    text = sys.argv[2]
    # track_amplitude(path, text, normalize=True)

    # track_rms(path, text)

    # track_rms_with_plot(path, text, block_size=512, normalize=True)
    track_rms_with_metadata(path, text, block_size=512, normalize=True)

if __name__ == "__main__":
    main()

# Example usage
# track_amplitude("input.wav", "amplitudes.txt", normalize=True)
