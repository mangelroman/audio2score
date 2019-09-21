import os
import sys
import argparse

import multiprocessing as mp
import subprocess
import csv
import json
import pickle

import numpy as np

from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from data.data_loader import load_audio
from data.humdrum import Kern, Labels, LabelsMulti

def parseList(string):
    if string and len(string) > 0:
        return string.split(',')
    return None

def parseIntList(string):
    if string and len(string) > 0:
        return [int(x) for x in string.split(',')]
    return None

parser = argparse.ArgumentParser(description='Spectrum preparation')
parser.add_argument('--data-dir', metavar='DIR', help='path to data', required=True)
parser.add_argument('--out-dir', metavar='DIR', help='path to output', required=True)
parser.add_argument('--min-duration-symbol', type=float, help='Select the minimum duration per symbol', required=True)
parser.add_argument('--max-duration', type=float, help='Select maximum duration of audio input files in seconds', required=True)
parser.add_argument('--num-workers', default=None, type=int, help='Number of workers used in data preparation')
parser.add_argument('--sample-rate', type=int, help='Select sampling frequency of WAV files', default=22050)
parser.add_argument('--bit-rate', type=int, help='Select bit rate of MP4 audio files', default=128000)
parser.add_argument('--soundfont', metavar='FILE', help='path to soundfont', default='/usr/share/sounds/sf2/FluidR3_GM.sf2')
parser.add_argument('--resynthesize', dest='resynthesize', action='store_true', default=False, help='Resynthesize audio from midi')
parser.add_argument('--instruments', type=parseList, help='Override kern defined intruments', default='piano')
parser.add_argument('--tempo-scaling', type=float, default=0.06, help='Select tempo random scaling')
parser.add_argument('--chunk-sizes', type=parseIntList, help='Select chunk sizes separated by commas', default=[sys.maxsize])
parser.add_argument('--test-split', type=float, default=0.3, help='Select train-test split ratio')
parser.add_argument('--train-stride', type=int, help='Select the stride of overlapped training samples', default=None)
parser.add_argument('--id', default='manifest', help='Id of output manifest and label files')
parser.add_argument('--labels-multi', action="store_true", default=False, help="Use multichar labels to reduce sequence size")

def process_sample(q, samples, args, labels):
    while True:
        score_path = q.get()
        if score_path is None:
            break

        # Remove grace notes, ornaments, etc...
        kern = Kern(Path(args.data_dir) / score_path)
        kern.spines.override_instruments(args.instruments)
        try:
            if not kern.clean():
                print(f'Cannot clean kern {score_path}')
                continue
        except Exception as e:
            print(f"Exception while cleaning {score_path} audio. Reason: {e}")
            continue

        root_path = Path(args.out_dir) / score_path.parent
        root_path.mkdir(parents=True, exist_ok=True)
        
        krn_path = Path(args.out_dir) / score_path

        # Set seed to ensure same chunk sizes and tempo scaling
        np.random.seed(bytearray(score_path.name, 'utf-8'))

        try:
            kern_chunks = kern.split(args.chunk_sizes, args.train_stride)
        except Exception as e:
            print(f'Exception {e} while splitting {score_path}')
            continue

        # random scale between +ts and -ts
        ts = 1 + args.tempo_scaling * (2 * np.random.rand(len(kern_chunks)) - 1)
        for i, kern in enumerate(kern_chunks):
            chunk_path = krn_path.with_suffix(f'.{i:03d}.krn')
            kern.save(chunk_path)

            # Fix ties with tiefix command
            process = subprocess.run(['tiefix', chunk_path], capture_output=True, encoding='iso-8859-1')
            if (process.returncode != 0):
                print(f"tiefix error={process.returncode} on {chunk_path}")
                print(process.stdout)
                continue

            kern = Kern(data=process.stdout)
            kern.save(chunk_path)

            audio_path = chunk_path.with_suffix('.flac')

            if args.resynthesize or not audio_path.exists():
                mid_path = chunk_path.with_suffix('.mid')
                # Tempo and instrumment extracted from *MM and *I indications
                status = os.system(f'hum2mid {str(chunk_path)} -C -v 100 -t {ts[i]} -o {str(mid_path)} >/dev/null 2>&1')
                if (os.WEXITSTATUS(status) != 0):
                    print(f"hum2mid error={status} on {krn_path}")
                    continue

                status = os.system(f'fluidsynth --sample-rate={args.sample_rate} -O s16 -T raw -i -l -F - {args.soundfont} {str(mid_path)} | '
                                   f'ffmpeg -y -f s16le -ar {args.sample_rate} -ac 2 -i pipe: '
                                   f'-ar {args.sample_rate} -ac 1 -ab {args.bit_rate} -strict -2 {str(audio_path)} 2>/dev/null')

            try:
                y = load_audio(str(audio_path))
            except Exception as e:
                print(f"Exception while loading {chunk_path} audio. Reason: {e}")
                continue

            duration = len(y) / args.sample_rate

            krnseq = kern.tosequence()

            if krnseq is None:
                #print(f"Discarded {chunk_path} for double dots/sharps/flats")
                continue

            try:
                seq = labels.encode(krnseq)
            except Exception as e:
                print(f"Discarded {chunk_path}. Reason: {e}")
                continue

            seqlen = labels.ctclen(seq)

            krnseq_path = chunk_path.with_suffix('.krnseq')
            krnseq_path.write_text(krnseq)

            seq_path = chunk_path.with_suffix('.seq')
            with seq_path.open(mode="wb") as f:
                f.write(pickle.dumps(seq))

            if duration > args.max_duration or duration < seqlen * args.min_duration_symbol:
                #print(f"Sequence too long in {chunk_path} len={seqlen} duration={duration:.2f}")
                continue

            samples.append([str(audio_path), str(seq_path), duration])

def process_scores(scores, args, labels):
    manager = mp.Manager()
    samples = manager.list()
    q = mp.Queue(maxsize=args.num_workers)
    pool = mp.Pool(args.num_workers, initializer=process_sample, initargs=(q, samples, args, labels))
    for score in tqdm(scores, ascii=True):
        if score.is_symlink():
            continue
        q.put(score)
    # stop workers
    for i in range(args.num_workers):
        q.put(None)
    pool.close()
    pool.join()

    x = np.array([x[0] for x in samples])
    y = np.array([x[1] for x in samples])
    durations = np.array([x[2] for x in samples])

    # SortaGrad
    sorted_indexes = np.argsort(durations)
    x = x[sorted_indexes]
    y = y[sorted_indexes]
    return x, y, np.sum(durations)

if __name__ == '__main__':
    args = parser.parse_args()

    print("Preprocessing humdrum data...")
    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    root = Path(args.data_dir)
    scores = sorted([x.relative_to(root) for x in root.rglob('*.krn')])

    print("Spliting train/test samples...")
    scores_train, scores_test = train_test_split(scores, test_size=args.test_split, random_state=45)

    middle = round(len(scores_test) / 2) # Favor validation if number of samples is odd
    scores_val = scores_test[:middle]
    scores_test = scores_test[middle:]

    labels = Labels() if not args.labels_multi else LabelsMulti()
    if args.num_workers is None:
        args.num_workers = 4

    print("Processing training samples:")
    x_train, y_train, train_dur = process_scores(scores_train, args, labels)

    # Force no overlap for validation and test samples
    args.train_stride = None

    print("Processing validation samples:")
    x_val, y_val, val_dur = process_scores(scores_val, args, labels)

    print("Processing test samples:")
    x_test, y_test, test_dur = process_scores(scores_test, args, labels)

    print("Number of train samples: {} ({:.2f} hours)".format(len(x_train), train_dur / 3600))
    print("Number of validation samples: {} ({:.2f} hours)".format(len(x_val), val_dur / 3600))
    print("Number of test samples: {} ({:.2f} hours)".format(len(x_test), test_dur / 3600))
    total_samples = len(x_train) + len(x_val) + len(x_test)
    total_dur = train_dur + val_dur + test_dur
    print("Total samples: {} ({:.2f} hours)".format(total_samples, total_dur / 3600))

    train_manifest_path = Path(f'train_{args.id}.csv')
    val_manifest_path = Path(f'val_{args.id}.csv')
    test_manifest_path = Path(f'test_{args.id}.csv')
    labels_path = Path(f'labels_{args.id}.json')

    with train_manifest_path.open(mode='w') as csvfile:
        writer = csv.writer(csvfile)
        print(f"Creating train manifest {train_manifest_path}...")
        for x, y in zip(x_train, y_train):
            writer.writerow([x, y])

    with val_manifest_path.open(mode='w') as csvfile:
        writer = csv.writer(csvfile)
        print(f"Creating val manifest {val_manifest_path}...")
        for x, y in zip(x_val, y_val):
            writer.writerow([x, y])

    with test_manifest_path.open(mode='w') as csvfile:
        writer = csv.writer(csvfile)
        print(f"Creating test manifest {test_manifest_path}...")
        for x, y in zip(x_test, y_test):
            writer.writerow([x, y])

    print("Creating label JSON file with {} symbols".format(len(labels.labels)))
    print(labels.labels)
    with labels_path.open(mode='w') as jsonfile:
        json.dump(labels.labels, jsonfile)
    
    sys.exit(0)
