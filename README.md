# audio2score

Implementation of audio-to-score task as referred in the paper "A holistic approach to polyphonic music transcription with neural networks" (link to follow).

The neural network architecture is based on [DeepSpeech2](https://arxiv.org/abs/1512.02595).

## Installation

Tested on Ubuntu Server 18.04 with miniconda package and environment manager.

1. Clone this repository and create a conda environment with the provided dependencies:
```
git clone https://github.com/mangelroman/audio2score.git
conda env create --file=audio2score/environment.yaml
conda activate a2s
```

2. Install pytorch audio:
```
sudo apt-get install sox libsox-dev libsox-fmt-all
git clone https://github.com/pytorch/audio.git
cd audio && python setup.py install
```

3. Install NVIDIA apex:
```
git clone --recursive https://github.com/NVIDIA/apex.git
cd apex && pip install .
```

4. Install Fluidsynth and FFmpeg
```
sudo apt-get install fluidsynth ffmpeg
```

5. Build Humdrum extra tools and copy them to a $PATH location 
```
git clone https://github.com/mangelroman/humextra.git
cd humextra
make library
make hum2mid
make tiefix
```


## Dataset

1. Create a folder to store all kern-based quartet repositories:
```
mkdir quartets && cd quartets
git clone https://github.com/mangelroman/humdrum-mozart-quartets.git
git clone https://github.com/mangelroman/humdrum-haydn-quartets.git
git clone https://github.com/mangelroman/beethoven-string-quartets.git
```
2. Run data preparation script pointing to the folder you just created
```
./prepquartets.sh id input_folder output_folder
```
The preparation script will create the following files in the current folder:
* train_id.csv with the training input and output file locations (stored in output_folder)
* val_id.csv with the validation input and output file locations (stored in output_folder)
* test_id.csv with the test input and output file locations (stored in output_folder)
* labels_id.json with the list of labels in the output representation

## Training

Run this shell script to start training with the default parameters:
```
./runtrain.sh config/quartets.cfg manifest_id model_id
```

If you want to run the training with other parameters, you may add them at the end of the previous command line, as in the following example:
```
./runtrain.sh config/quartets.cfg manifest_id model_id --num-workers 4
```
Please run ```python train.py --help``` for a complete list of options.

You may also check the [configuration file](config/quartets.cfg) for extended training parameters.

### Multi-GPU

Add -m multiproc to the training script as in the following snipet:
```
python -m multiproc train.py --cuda  # Add your parameters as normal
```

### Mixed Precision

```
python train.py --cuda --mixed-precision # Add your parameters as normal
```
Mixed precision can also be combined with multi-GPU:
```
python -m multiproc train.py --cuda --mixed-precision  # Add your parameters as normal
```

### Checkpoints

To continue from a checkpointed model that has been saved:

```
./runtrain.sh config/quartets.cfg manifest_id model_id --continue-from models/model_id.last.pth
```

If you would like to start from a previous checkpoint model but not continue training, add the `--finetune` flag to restart training from the `--continue-from` model state.

After each epoch a new checkpoint is always saved with one of these namings:
```
models/model_id.pth # For the best model so far
models/model_id.last.pth # For the last model that is not the best so far
```

## Testing

To evaluate a trained model on a test set:

```
./runtest.sh test_id.csv models/model_id.pth
```

An example script to output a single transcription has been provided:
```
python transcribe.py --model-path models/model_id.pth --audio-path /path/to/audio.wav
```

## Model

To display hyperparams and other training metadata of any checkpointed model:

```
python showmodel.py models/model_id.pth
```

## Acknowledgements

Most of this code is borrowed from [Sean Naren](https://github.com/SeanNaren/deepspeech.pytorch).