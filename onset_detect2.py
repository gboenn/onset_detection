import sys
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import LogLocator
from scipy.signal import find_peaks
import mido
import csv

def analyze_novelty_with_peaks(input_file, block_size=1024, C=1000, 
                              avg_window=5, peak_thresh=0.1, peak_distance=0.1,
                              low_freq_thresh=20, high_freq_thresh=None,
                              output_peaks="onsets.txt", output_clicks="detected_clicks.wav", output_tones="frequency_tones.wav", output_midi="frequency_tones.mid"):
    """Compute and visualize spectrogram with novelty features and onset peak detection according to Grosche and Mueller"""
    
    # Load and process audio
    audio, sample_rate = sf.read(input_file, always_2d=True)
    audio_mono = np.mean(audio, axis=1)
    
    if np.issubdtype(audio_mono.dtype, np.integer):
        audio_mono = audio_mono.astype(np.float32) / np.iinfo(audio_mono.dtype).max

    # STFT parameters
    window = np.hanning(block_size)
    hop_size = block_size // 3
    n_blocks = (len(audio_mono) - block_size) // hop_size + 1
    
    # Generate frequency edges
    freqs = np.fft.rfftfreq(block_size, 1/sample_rate)
    freq_edges = np.zeros(len(freqs) + 1)
    freq_edges[0] = 0
    freq_edges[1:-1] = (freqs[:-1] + freqs[1:]) / 2
    freq_edges[-1] = freqs[-1]
    
    # Time edges for alignment
    time_edges = np.arange(n_blocks + 1) * hop_size / sample_rate
    delta_time = (time_edges[1:-1] + time_edges[2:])/2

    # Compute compressed spectrogram with frequency filtering
    spec_compressed = np.zeros((len(freqs), n_blocks))
    for i in range(n_blocks):
        start = i * hop_size
        block = audio_mono[start:start+block_size] * window
        spectrum = np.fft.rfft(block / block_size)
        
        # Apply frequency filtering
        filtered_spectrum = np.copy(spectrum)
        filtered_spectrum[freqs < low_freq_thresh] = 0
        if high_freq_thresh is not None:
            filtered_spectrum[freqs > high_freq_thresh] = 0
        
        spec_compressed[:,i] = np.log1p(C * np.abs(filtered_spectrum))

    # Compute novelty features
    delta = np.sum(np.abs(np.diff(spec_compressed, axis=1)), axis=0)
    delta_avg = np.convolve(delta, np.ones(avg_window)/avg_window, 'same')
    delta_diff = np.maximum(delta - delta_avg, 0)
    delta_diff_norm = (delta_diff - np.min(delta_diff)) / (np.max(delta_diff) - np.min(delta_diff))

    # Find peaks in normalized residuals
    min_peak_distance = int(peak_distance * sample_rate / hop_size)
    peaks, _ = find_peaks(delta_diff_norm, 
                         height=peak_thresh,
                         distance=min_peak_distance,
                         prominence=0.05)
    peak_times = delta_time[peaks]
    peak_values = delta_diff_norm[peaks]

    # Generate click track
    click_sample_rate = 48000
    click_samples = int(round(len(audio_mono) / sample_rate * click_sample_rate))
    click_track = np.zeros(click_samples)
    for pt in peak_times:
        sample_idx = int(round(pt * click_sample_rate))
        if 0 <= sample_idx < click_samples:
            click_track[sample_idx] = 0.5  # -6 dBFS

    # Create MIDI file
    ppq = 960
    tpo = int(60000000. / 120)
    mid = mido.MidiFile(ticks_per_beat=ppq)
    track = mido.MidiTrack()
    track.append(mido.MetaMessage('set_tempo', tempo=tpo, time=0))
    mid.tracks.append(track)
    ###########################
    # Generate frequency-tuned tone track
    tone_sample_rate = 48000
    tone_duration = int(0.1 * tone_sample_rate)  # 100ms tone
    tone_track = np.zeros(int(round(len(audio_mono) / sample_rate * tone_sample_rate)))

    nfreqs = 8
    # for pt in peak_times:
    #     # Find strongest three frequencies at onset time
    #     onset_idx = np.argmin(np.abs(delta_time - pt))
    #     strongest_freqs_idx = np.argsort(spec_compressed[:,onset_idx])[-(nfreqs):]  # Last three indices are strongest
    #     strongest_freqs = freqs[strongest_freqs_idx]
        
    #     # Generate windowed sine tones for each frequency
    #     tone_start = int(round(pt * tone_sample_rate))
    #     tone_end = tone_start + tone_duration
    #     if tone_end <= len(tone_track):
    #         t = np.linspace(0, tone_duration/tone_sample_rate, tone_duration)
    #         for freq in strongest_freqs:
    #             tone = np.sin(2 * np.pi * freq * t) * np.hanning(tone_duration)
    #             tone_track[tone_start:tone_end] += tone / nfreqs  # Normalize amplitude to avoid clipping

######################
    # Quantization grid in milliseconds
    # quantization_grid_ms = 2
    midi_time = 0
    delta_midi = 0
    midi_note = 0
    midi_velocity = 0
    # first_note = 1
    # ten_ms = mido.second2tick(0.1, ppq, tpo)
    for pt in peak_times:
        # Find strongest three frequencies at onset time
        onset_idx = np.argmin(np.abs(delta_time - pt))
        strongest_freqs_idx = np.argsort(spec_compressed[:,onset_idx])[-(nfreqs):]  # Last three indices are strongest
        strongest_freqs = freqs[strongest_freqs_idx]
        strongest_amplitudes = spec_compressed[strongest_freqs_idx, onset_idx]
        
        # Normalize amplitudes to ensure they sum to 1 (for consistent overall level)
        normalized_amplitudes = strongest_amplitudes / np.sum(strongest_amplitudes)
        
        # Generate windowed sine tones for each frequency
        tone_start = int(round(pt * tone_sample_rate))
        tone_end = tone_start + tone_duration
        midi_time = pt - delta_midi

        if tone_end <= len(tone_track):
            t = np.linspace(0, tone_duration/tone_sample_rate, tone_duration)
            first_note = 1
            for freq, amp in zip(strongest_freqs, normalized_amplitudes):
                tone = np.sin(2 * np.pi * freq * t) * np.hanning(tone_duration) * amp
                tone_track[tone_start:tone_end] += tone
                # Convert frequency to MIDI note
                midi_note = int(round(69 + 12 * np.log2(freq / 440)))  # A4 = 440 Hz
                 # Scale amplitude to MIDI velocity (0-127)
                midi_velocity = 0
                if (amp >= 0. and amp < 1.):
                    midi_velocity = int(round(amp * 127))
                
                                
                # Add MIDI note-on and note-off events    
                if (first_note == 1): 
                    track.append(mido.Message('note_on', note=midi_note, velocity=midi_velocity, time=int(mido.second2tick(midi_time, ppq, tpo))))  # Adjusted time
                    first_note = 0
                else:
                    track.append(mido.Message('note_on', note=midi_note, velocity=midi_velocity))
                # track.append(mido.Message('note_off', note=midi_note, velocity=midi_velocity, time=int(mido.second2tick(midi_time+0.1, ppq, tpo))))

        delta_midi = pt
    # last note off
    track.append(mido.Message('note_off', note=midi_note, velocity=midi_velocity, time=int(mido.second2tick(midi_time, ppq, tpo))))

    # Save outputs
    # np.savetxt(output_peaks, peak_times, fmt='%.6f', 
            #   header=f"Peak times (seconds)\n"
                    #  f"Threshold: {peak_thresh}, Distance: {peak_distance}s")
    
    with open(output_peaks, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for pkt in peak_times:
            writer.writerow([pkt])

    sf.write(output_clicks, click_track, click_sample_rate, subtype='PCM_16')
    sf.write(output_tones, tone_track, tone_sample_rate, subtype='PCM_16')
    mid.save(output_midi)
    # Visualization code remains unchanged...
    
    # Create 3-row layout
    plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 1, height_ratios=[3, 1, 1], hspace=0.4)
    
    # Spectrogram plot
    ax0 = plt.subplot(gs[0])
    mesh = ax0.pcolormesh(time_edges, freq_edges, spec_compressed,
                         shading='auto', cmap='inferno',
                         rasterized=True)
    # plt.colorbar(mesh, ax=ax0, label=f'log(1 + {C}·|X|)')
    ax0.set_yscale('symlog', linthresh=1000)
    ax0.set_ylim(low_freq_thresh, sample_rate//2 if high_freq_thresh is None else high_freq_thresh)
    ax0.set_title(f"Compressed Spectrogram: {input_file}\n"
                 f"Block Size: {block_size}, SR: {sample_rate/1000:.1f} kHz")
    
    # Rest of plotting code remains unchanged...
    # Original novelty plot (unchanged)
    ax1 = plt.subplot(gs[1], sharex=ax0)
    # ... [identical novelty plotting code] ...
    ax1.plot(delta_time, delta, color='#ff7f0e', linewidth=0.8, alpha=0.7, label='Novelty')
    ax1.plot(delta_time, delta_avg, color='#2ca02c', linewidth=1.5, label=f'{avg_window}-Frame Average')
    ax1.fill_between(delta_time, 0, delta, color='#ff7f0e', alpha=0.15)
    ax1.legend(loc='upper right')

    # Enhanced residual plot with peaks
    ax2 = plt.subplot(gs[2], sharex=ax0)
    ax2.plot(delta_time, delta_diff_norm, color='#1f77b4', linewidth=0.8, label='Normalized Residuals')
    ax2.fill_between(delta_time, 0, delta_diff_norm, color='#1f77b4', alpha=0.3)
    ax2.scatter(peak_times, peak_values, color='red', s=20, zorder=3, 
               label=f'Peaks (>{peak_thresh*100:.0f}%)')
    
    # Formatting
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Normalized Residuals')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')
    
    # Set y-axis limits for residual plot
    ax2.set_ylim(-0.05, 1.05)
    
    # Draw threshold line
    ax2.axhline(peak_thresh, color='gray', linestyle='--', alpha=0.7, linewidth=0.8)
    
    # Apply consistent time limits
    time_min = max(time_edges[0], delta_time[0])
    time_max = min(time_edges[-1], delta_time[-1])
    ax0.set_xlim(time_min, time_max)
    
    plt.tight_layout()
    plt.savefig('novelty_peaks.png', dpi=150, bbox_inches='tight')
    plt.close()


def main():
    if len(sys.argv) < 8:
        print("Usage: python onset_detector.py input_file block_size C avg_window peak_thresh low_freq_thresh [high_freq_thresh]")
        return

    path = sys.argv[1]
    blocksize = int(sys.argv[2])
    comp = int(sys.argv[3])
    nwin = int(sys.argv[4])
    pthresh = float(sys.argv[5])
    low_freq = float(sys.argv[6])
    high_freq = float(sys.argv[7]) if len(sys.argv) > 7 else None

    analyze_novelty_with_peaks(path, blocksize, comp, nwin, pthresh, 0.05, low_freq, high_freq)

if __name__ == "__main__":
    main()
