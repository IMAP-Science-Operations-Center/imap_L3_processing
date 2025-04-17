#!/bin/bash

# validate L3e by comparison of files in the directory L3e_output_from_M_Kubiak_for_validation
# with files generated locally in the docker by the script run_example_L3d_to_L3e.sh

# Only lines containing "Date of calculation:" are expected to be different if
# the validation is successful

for item_by_MK in `ls L3e_output_from_M_Kubiak_for_validation/probSur.Imap.*.dat`; do
    item=`basename $item_by_MK`
    echo "$item_by_MK $item"
    diff $item_by_MK $item | head -20
    echo
    echo "#########################################################################################"
done
