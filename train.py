import argparse
import configparser
import json
import os
import random
import time
import csv

from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.utils.data.distributed
from apex.parallel import DistributedDataParallel
from apex import amp
from tqdm import tqdm

from data.data_loader import AudioDataLoader, SpectrogramDataset, BucketingSampler, DistributedBucketingSampler
from utils import AverageMeter, LabelDecoder, calculate_wer, calculate_cer, calculate_ler, load_model, save_model

parser = argparse.ArgumentParser(description='DeepSpeech training')
parser.add_argument('--train-manifest', metavar='DIR', required=True, help='path to train manifest csv')
parser.add_argument('--val-manifest', metavar='DIR', required=True, help='path to validation manifest csv')
parser.add_argument('--labels-path', metavar='DIR', required=True, help='Contains all characters for transcription')
parser.add_argument('--config-path', metavar='DIR', required=True, help='path to configuration ini')
parser.add_argument('--continue-from', metavar='DIR', help='Continue from checkpoint model')
parser.add_argument('--model-path', metavar='DIR', required=True, help='Location to save best validation model')
parser.add_argument('--num-workers', default=4, type=int, help='Number of workers used in data-loading')
parser.add_argument('--cuda', dest='cuda', action='store_true', help='Use cuda to train model')
parser.add_argument('--silent', dest='silent', action='store_true', help='Turn off progress tracking per iteration')
parser.add_argument('--finetune', dest='finetune', action='store_true',
                    help='Finetune the model from checkpoint "continue_from"')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:1550', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--rank', default=0, type=int,
                    help='The rank of this process')
parser.add_argument('--gpu-rank', default=None,
                    help='If using distributed parallel for multi-gpu, sets the GPU for the process')
parser.add_argument('--seed', default=45, type=int, help='Seed to generators')
parser.add_argument('--mixed-precision', action='store_true',
                    help='Uses mixed precision to train a model (suggested with volta and above)')

if __name__ == '__main__':
    args = parser.parse_args()
    config = configparser.ConfigParser()
    config.read(args.config_path)
    model_name = config['train']['model']
    model_conf = config[model_name]
    audio_conf = config['audio']

    if model_name == "deepspeech":
        from deepspeech.model import DeepSpeech as Model
        from deepspeech.loss import Loss
    else:
        raise NotImplementedError

    train_conf = config['train']
    batch_size = train_conf.getint('batch_size')
    epochs = train_conf.getint('epochs')
    shuffle = train_conf.getboolean('shuffle', True)
    sorta_grad = train_conf.getboolean('sorta_grad', True)

    # Set seeds for determinism
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.mixed_precision and not args.cuda:
        raise ValueError('If using mixed precision training, CUDA must be enabled!')

    args.distributed = args.world_size > 1
    main_proc = True
    device = torch.device("cuda" if args.cuda else "cpu")
    if args.distributed:
        if args.gpu_rank:
            torch.cuda.set_device(int(args.gpu_rank))
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        main_proc = args.rank == 0  # Only the first proc should save models

    save_folder = os.path.dirname(args.model_path)
    os.makedirs(save_folder, exist_ok=True)  # Ensure save folder exists

    train_results, val_results = torch.Tensor(epochs), torch.Tensor(epochs)
    best_wer = None
    last_model_path = Path(args.model_path).with_suffix('.last.pth')

    results_path = Path(args.model_path).with_suffix('.csv')
    with open(results_path, 'a') as resfile:
        wr = csv.writer(resfile)
        wr.writerow(['Epoch', 'Train Loss', 'Val WER', 'Val CER', 'Val LER', 'Train Time', 'Val Time'])

    train_loss, start_epoch, optim_state = 0, 0, None
    if args.continue_from:  # Starting from previous model
        print("Loading checkpoint model %s" % args.continue_from)
        model, package = load_model(args.continue_from)
        labels = model.labels
        audio_conf = model.audio_conf
        model_conf = model.model_conf
        if not args.finetune:  # Don't want to restart training
            optim_state = package['optim_dict']
            start_epoch = int(package.get('epoch', 0)) + 1  # Index start at 0 for training
            train_loss = int(package.get('avg_loss', 0))
            for i in range(start_epoch):
                train_results[i] = package['train_results'][i]
                val_results[i] = package['val_results'][i]
            best_wer = float(val_results[:start_epoch].min())
    else:
        with open(args.labels_path) as label_file:
            labels = json.load(label_file)

        model = Model(model_conf, audio_conf, labels)

    model = model.to(device)
    train_dataset = SpectrogramDataset(audio_conf=audio_conf, manifest_filepath=args.train_manifest, labels=labels)
    val_dataset = SpectrogramDataset(audio_conf=audio_conf, manifest_filepath=args.val_manifest, labels=labels)
    label_decoder = LabelDecoder(labels)
    
    if not args.distributed:
        train_sampler = BucketingSampler(train_dataset, batch_size=batch_size)
        val_sampler = BucketingSampler(val_dataset, batch_size=batch_size)
    else:
        train_sampler = DistributedBucketingSampler(train_dataset, batch_size=batch_size,
                                                    num_replicas=args.world_size, rank=args.rank)
        val_sampler = DistributedBucketingSampler(val_dataset, batch_size=batch_size,
                                                    num_replicas=args.world_size, rank=args.rank)

    train_loader = AudioDataLoader(train_dataset, num_workers=args.num_workers, batch_sampler=train_sampler)
    val_loader = AudioDataLoader(val_dataset, num_workers=args.num_workers, batch_sampler=val_sampler)

    if (shuffle and start_epoch != 0) or not sorta_grad:
        print("Shuffling batches for the following epochs")
        train_sampler.shuffle(start_epoch)
    val_sampler.shuffle(1)

    optim_name = train_conf['optimizer']
    optim_conf = config[optim_name]
    parameters = model.parameters()
    learning_rate = train_conf.getfloat('learning_rate')
    max_norm = train_conf.getfloat('max_norm')
    if optim_name == 'rmsprop':
        optimizer = torch.optim.RMSprop(parameters, lr=learning_rate, alpha=optim_conf.getfloat('alpha', 0.95),
                                        eps=optim_conf.getfloat('epsilon', 1e-8))
    elif optim_name == 'adam':
        betas = [optim_conf.getfloat('beta1', 0.9), optim_conf.getfloat('beta2', 0.999)]
        optimizer = torch.optim.Adam(parameters, lr=learning_rate, betas=betas)
    elif optim_name == 'sgd':
        optimizer = torch.optim.SGD(parameters, lr=learning_rate, momentum=optim_conf.getfloat('momentum', 0.9), 
                                    nesterov=optim_conf.getboolean('nesterov', True))
    else:
        raise NotImplementedError

    if not args.mixed_precision:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O0')
    else:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

    if args.distributed:
        model = DistributedDataParallel(model, delay_allreduce=True)
        modelbase = model.module
    else:
        modelbase = model

    print(model)
    print("Number of parameters: %d" % Model.get_param_size(model))

    if optim_state is not None:
        optimizer.load_state_dict(optim_state)

    criterion = Loss(model, device)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=train_conf.getfloat('learning_anneal'))

    batch_time = AverageMeter()
    data_time = AverageMeter()
    train_losses = AverageMeter()

    for epoch in range(start_epoch, epochs):
        model.train()
        end = time.time()
        start_epoch_time = time.time()
        for i, (data) in enumerate(train_loader):
            inputs, targets, input_sizes, target_sizes, filenames = data
            # measure data loading time
            data_time.update(time.time() - end)

            loss_value = 0
            try:
                loss = criterion.calculate_loss(inputs, input_sizes, targets, target_sizes)
            except Exception as error:            
                print(error)
                print('Skipping grad update')
            else:
                optimizer.zero_grad()
                # compute gradient
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                # clip gradients
                if max_norm:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), max_norm)
                # update parameters
                optimizer.step()
                if args.distributed:
                    dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                    loss_value = loss.item() / args.world_size
                else:
                    loss_value = loss.item()

            train_loss += loss_value
            train_losses.update(loss_value, batch_size)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if not args.silent:
                print(f'Epoch: [{epoch + 1}][{i + 1}/{len(train_sampler)}]\tTime {batch_time.value:.3f} ({batch_time.avg:.3f})\t'
                      f'Data {data_time.value:.3f} ({data_time.avg:.3f})\tLoss {train_losses.value:.4f} ({train_losses.avg:.4f})\t')
        train_loss /= len(train_sampler)

        epoch_time = time.time() - start_epoch_time
        print(f'Training Summary Epoch: [{epoch + 1}]\tTime taken (s): {epoch_time:.0f}\tAverage Loss {train_loss:.3f}\t')

        val_wer, val_cer, val_ler, num_words, num_chars, num_labels = 0, 0, 0, 0, 0, 0
        model.eval()
        with torch.no_grad():
            for i, (data) in tqdm(enumerate(val_loader), total=len(val_loader), ascii=True):
                inputs, targets, input_sizes, target_sizes, filenames = data
                inputs = inputs.to(device)
                input_sizes = input_sizes.to(device)
                transcripts = modelbase.transcribe(inputs, input_sizes)
                for i, target in enumerate(targets):
                    reference = label_decoder.decode(target[:target_sizes[i]].tolist())
                    transcript = label_decoder.decode(transcripts[i])
                    wer, trans_words, ref_words = calculate_wer(transcript, reference, '\t')
                    cer, trans_chars, ref_chars = calculate_cer(transcript, reference, '\t')
                    ler, trans_labels, ref_labels = calculate_ler(transcript, reference)
                    val_wer += wer
                    num_words += ref_words
                    val_cer += cer
                    num_chars += ref_chars
                    val_ler += ler
                    num_labels += ref_labels

            if args.distributed:
                result_tensor = torch.tensor([val_wer, val_cer, val_ler, num_words, num_chars, num_labels]).to(device)
                dist.all_reduce(result_tensor, op=dist.ReduceOp.SUM)
                val_wer, val_cer, val_ler, num_words, num_chars, num_labels = result_tensor
            val_wer = 100 * float(val_wer) / num_words
            val_cer = 100 * float(val_cer) / num_chars
            val_ler = 100 * float(val_ler) / num_labels
            val_time = time.time() - start_epoch_time - epoch_time
            print(f'Validation Summary Epoch: [{epoch + 1}]\tTime taken (s): {val_time:.0f}\t'
                  f'Average WER {val_wer:.3f}\tAverage CER {val_cer:.3f}\tAverage LER {val_ler:.3f}')
        if main_proc:
            train_results[epoch] = train_loss
            val_results[epoch] = val_wer
            with open(results_path, 'a') as resfile:
                wr = csv.writer(resfile)
                wr.writerow([epoch + 1, train_loss, val_wer, val_cer, val_ler, int(epoch_time), int(val_time)])

            if best_wer is None or best_wer > val_wer:
                print("Found better validated model, saving to %s" % args.model_path)
                save_model(modelbase, args.model_path, optimizer=optimizer, epoch=epoch,
                            train_results=train_results, val_results=val_results)
                best_wer = val_wer
            else:
                save_model(modelbase, str(last_model_path), optimizer=optimizer, epoch=epoch,
                            train_results=train_results, val_results=val_results)
        train_loss = 0
        scheduler.step()
        if shuffle:
            print("Shuffling batches...")
            train_sampler.shuffle(epoch)
