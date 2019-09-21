if [[ $# -lt 2 ]] ; then
	echo 'missing arguments: ./runtest.sh manifest_path model_path [args]'
	exit -1
fi
manifest=$1
model=$2
shift 2
python test.py --cuda --test-manifest $manifest --batch-size 20 --model-path $model --verbose $*
