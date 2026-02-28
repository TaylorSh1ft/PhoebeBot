#!/usr/bin/env python3
import subprocess, os

os.makedirs('/home/phoebe/tv_negatives', exist_ok=True)

for i in range(1, 51):
    fname = f'/home/phoebe/tv_negatives/sp_{i:02d}.wav'
    print(f'clip {i}/50...', flush=True)
    subprocess.run([
        'arecord', '-D', 'plughw:2,0',
        '-r', '16000', '-f', 'S16_LE', '-c', '1',
        '-d', '5', fname
    ], stderr=subprocess.DEVNULL)
    print(f'  saved', flush=True)

print('RECORDING_COMPLETE')
