import os
import argparse

import torch

from utils import load_model

parser = argparse.ArgumentParser(description='DeepSpeech model information')
parser.add_argument('model_path', metavar='FILE', help='Path to model created by training')
args = parser.parse_args()

if __name__ == '__main__':

    model, package = load_model(args.model_path)

    print("Model name:    ", model.name)
    print("Model version: ", model.version)
    print("")
    print(model)
    print("")
    print("Model Features")
    print("  Input Format:     ", model.audio_conf.get("input_format", "n/a"))
    print("  Sample Rate:      ", model.audio_conf.get("sample_rate", "n/a"))
    print("  Window Type:      ", model.audio_conf.get("window", "n/a"))
    print("  Window Size:      ", model.audio_conf.get("window_size", "n/a"))
    print("  Window Stride:    ", model.audio_conf.get("window_stride", "n/a"))
    print("  Minimum Note:     ", model.audio_conf.get("min_note", "n/a"))
    print("  Bins per octave:  ", model.audio_conf.get("bins_per_octave", "n/a"))
    print("  Number of octaves:", model.audio_conf.get("num_octaves", "n/a"))
    print("  Output Classes:   ", len(model.labels))
    print("  Labels:           ")
    print(model.labels)
    print("")

    if package.get('optim_dict', None) is not None:
        print("Optimizer")
        print(f"    Params: {package['optim_dict']['param_groups'][0].keys()}")
        try:
            print(f"    LR_init: {package['optim_dict']['param_groups'][0]['initial_lr']}")
        except KeyError:
            pass
        print(f"    LR_current: {package['optim_dict']['param_groups'][0]['lr']}")
        print("")
    if package.get('train_results', None) is not None:
        print("Training Information")
        epochs = package['epoch']
        print("  Epochs:           ", epochs + 1)
        print("  Current Train Results:      {0:.3f}".format(package['train_results'][epochs]))
        print("  Current Validation Results: {0:.3f}".format(package['val_results'][epochs]))