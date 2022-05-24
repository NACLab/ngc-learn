#!/bin/bash

# runs a master-level execution of ALL bash scripts contained within each
# demo folder

# WARNING: this will execute ALL of the demo files in order, so beware that this
# will take some time to run. All demo code should run without breaking assuming
# that you have unzipped all of the ZIP files w/in the /demos/data/ sub-directory
# and placed the resulting folders of data arrays exactly in /demos/data/
#
# You can run this (for a more qualitative global check) all of the demos/tutorials
# with the following master execution script (but you will HAVE to currently
# check the outputs of each demonstration in their respective folder matches
# what appears in the tutorials themselves)

echo "-> Executing ALL walkthrough scripts!"
cd ../walkthroughs/ # enter demo directoy, start running demo scripts
echo "Running Demo 1"
cd walkthrough1/
./exec_experiments.sh
cd ../
echo "Running Demo 2"
cd walkthrough2/
./exec_experiments.sh
cd ../
echo "Running Demo 3"
cd walkthrough3/
./exec_experiments.sh
cd ../
echo "Running Demo 4"
cd walkthrough4/
./exec_experiments.sh
cd ../
echo "Running Demo 5"
cd walkthrough5/
./exec_experiments.sh
cd ../
echo "Running Demo 6"
cd walkthrough6/
./exec_experiments.sh
cd ../../ # leave demo directoy, we are done!
echo "-> Master walkthrough test completed!"
