import argparse

import numpy as np
import torch

from tqdm import tqdm

from data.data_loader import SpectrogramDataset, AudioDataLoader, BucketingSampler
from utils import LabelDecoder, calculate_wer, calculate_cer, calculate_ler, load_model

parser = argparse.ArgumentParser(description='DeepSpeech transcription')
parser.add_argument('--cuda', action="store_true", help='Use cuda to test model')
parser.add_argument('--model-path', default='models/model_default.pth',
                    help='Path to model file created by training')
parser.add_argument('--test-manifest', metavar='DIR',
                    help='path to validation manifest csv', default='data/test_manifest.csv')
parser.add_argument('--batch-size', default=20, type=int, help='Batch size for training')
parser.add_argument('--num-workers', default=4, type=int, help='Number of workers used in dataloading')
parser.add_argument('--verbose', action="store_true", help="print out decoded output and error of each sample")
parser.add_argument('--output-path', default=None, type=str, help="Where to save raw acoustic output")
args = parser.parse_args()

if __name__ == '__main__':
    torch.set_grad_enabled(False)
    model, _ = load_model(args.model_path)
    device = torch.device("cuda" if args.cuda else "cpu")
    label_decoder = LabelDecoder(model.labels)
    model.eval()
    model = model.to(device)

    test_dataset = SpectrogramDataset(audio_conf=model.audio_conf, manifest_filepath=args.test_manifest,
                                      labels=model.labels)
    test_sampler = BucketingSampler(test_dataset, batch_size=args.batch_size)
    test_loader = AudioDataLoader(test_dataset, batch_sampler=test_sampler,
                                  num_workers=args.num_workers)
    test_sampler.shuffle(1)

    total_wer, total_cer, total_ler, num_words, num_chars, num_labels = 0, 0, 0, 0, 0, 0
    output_data = []

    for i, (data) in tqdm(enumerate(test_loader), total=len(test_loader), ascii=True):
        inputs, targets, input_sizes, target_sizes, filenames = data
        inputs = inputs.to(device)
        input_sizes = input_sizes.to(device)
        outputs = model.transcribe(inputs, input_sizes)

        for i, target in enumerate(targets):
            reference = label_decoder.decode(target[:target_sizes[i]].tolist())
            transcript = label_decoder.decode(outputs[i])
            wer, trans_words, ref_words = calculate_wer(transcript, reference, '\t')
            cer, trans_chars, ref_chars = calculate_cer(transcript, reference, '\t')
            ler, trans_labels, ref_labels = calculate_ler(transcript, reference)
            total_wer += wer
            num_words += ref_words
            total_cer += cer
            num_chars += ref_chars
            total_ler += ler
            num_labels += ref_labels

            if args.verbose:
                print("File:", filenames[i])
                print("WER:", float(wer) / ref_words)
                print("CER:", float(cer) / ref_chars)
                print("LER:", float(ler) / ref_labels)
                print("========================================================= REFERENCE:")
                print(reference)
                print("========================================================= HYPOTHESIS:")
                print(transcript)
                print("")

    wer = 100 * float(total_wer) / num_words
    cer = 100 * float(total_cer) / num_chars
    ler = 100 * float(total_ler) / num_labels

    print(f'Test Summary \tAverage WER {wer:.3f}\tAverage CER {cer:.3f}\tAverage LER {ler:.3f}')
