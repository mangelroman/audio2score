if [[ $# -lt 3 ]] ; then
    echo 'missing arguments: ./runtrain.sh config manifest_id model_id [args]'
    exit -1
fi
config=$1
manid=$2
modid=$3
shift 3
# Or python -m multiproc for multi-GPU training 
python train.py --cuda --config-path ${config} --train-manifest train_${manid}.csv --val-manifest val_${manid}.csv --labels-path labels_${manid}.json --num-workers 4 --model-path models/${modid}.pth $*
