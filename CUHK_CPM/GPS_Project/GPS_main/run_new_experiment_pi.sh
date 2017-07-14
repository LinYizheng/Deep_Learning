#!/bin/bash -x

 _predix="ja_mode"
#_predix="TrajOpt"
_now="`date +%m%d%Y_%H%M%S`"
_file="${_predix}_${_now}.experiment"
python python/gps/gps_main.py ${_file} -n

HyperparametersFILE="experiments/${_file}/hyperparams.py"

# cp experiments/pr2_badmm_example/hyperparams.py $HyperparametersFILE

sed -i "s/^EXP_DIR.*/EXP_DIR = BASE_DIR + '\/..\/experiments\/${_file}\/'/" $HyperparametersFILE
# echo 'haha'
# python python/gps/gps_main.py ${_file} -t

