# Command-line:

> python onset_detector.py input_file block_size C avg_window peak_thresh low_freq_thresh high_freq_thresh

input_file: mono or stereo wav file 
block_size: STFT window size, e.g. 512, 1024, 2048, ...
C: compression factor, typical values are 5000 or higher
avg_window: number of windows to average over, typically a small integer like 3 - 7 windows
peak_thresh: a threshold percentage for peak picking, e.g. 0.25 
low_freq_thresh: low frequency threshold in Hz > 0. 
high_freq_thresh: high frequency threshold in Hz
The spectral analysis output can be filtered between low and high frequency thresholds.

# Example:

> python3 onset_detect2.py example.wav 512 8000 7 0.25 1 4000

# Output files:

> onsets.txt

list of all detected onsets in seconds

> novelty_peaks.png

plot of the spectrogram and a series of stages that are part the onset detection process

> detected_clicks.wav

audio click track of the onsets at -6dB

> frequency_tones.wav

synthesized tones of the strongest peaks in the audio spectrum at the time of onset

> frequency_tones.mid

same as frequency_tones.wav with MIDI notes for each spectral peak




