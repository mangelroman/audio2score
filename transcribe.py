import os
import argparse

import torch

from data.data_loader import SpectrogramParser
from utils import load_model, LabelDecoder

parser = argparse.ArgumentParser(description='DeepSpeech transcription')
parser.add_argument('--cuda', action="store_true", default=False, help='Use cuda to test model')
parser.add_argument('--model-path', help='Path to model file created by training', required=True)
parser.add_argument('--audio-path', help='Audio file to predict on', required=True)
args = parser.parse_args()

if __name__ == '__main__':
    model, _ = load_model(args.model_path)
    device = torch.device("cuda" if args.cuda else "cpu")
    label_decoder = LabelDecoder(model.labels)
    model.eval()
    model = model.to(device)

    parser = SpectrogramParser(model.audio_conf)
    spect = parser.parse_audio(args.audio_path).contiguous()
    spect = spect.view(1, spect.size(0), spect.size(1))
    spect = spect.to(device)
    input_sizes = torch.IntTensor([spect.size(1)])
    input_sizes.to(device)

    transcripts = model.transcribe(spect, input_sizes)
    print(label_decoder.decode(transcripts[0]))