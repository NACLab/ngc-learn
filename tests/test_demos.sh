#!/bin/bash

# runs a master-level execution of ALL bash scripts contained within each
# demo folder

# WARNING: this will execute ALL of the demo files in order, so beware that this
# will take some time to run. All demo code should run without breaking assuming
# that you have unzipped all of the ZIP files w/in the /examples/data/ sub-directory
# and placed the resulting folders of data arrays exactly in /examples/data/

echo "-> Executing ALL demonstration scripts!"
cd ../examples/ # enter demo directoy, start running demo scripts
echo "Running Demo 1"
cd demo1/
./exec_experiments.sh
cd ../
echo "Running Demo 2"
cd demo2/
./exec_experiments.sh
cd ../
echo "Running Demo 3"
cd demo3/
./exec_experiments.sh
cd ../
echo "Running Demo 4"
cd demo4/
./exec_experiments.sh
cd ../
echo "Running Demo 5"
cd demo5/
./exec_experiments.sh
cd ../../ # leave demo directoy, we are done!
echo "-> Master demo test completed!"
