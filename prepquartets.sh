if [[ $# -lt 3 ]] ; then
    echo 'missing arguments: ./prepdata.sh id datadir outdir [args]'
    exit -1
fi
id=$1
datadir=$2
outdir=$3
shift 3
python prepare.py --data-dir $datadir --out-dir $outdir --num-workers 8 --min-duration-symbol 0.01161 --max-duration 30.0 --test-split 0.3 --id $id --instruments cello,viola,violn,flt --tempo-scaling 0.06 --chunk-sizes 3,4,5,6 --labels-multi --train-stride 1 $*
