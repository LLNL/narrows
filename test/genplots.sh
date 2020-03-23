# genplots.sh

# Run p1 and generate plots
echo "$0: Running p1"
./p1-full_slab.py
echo "$0: Generating p1 analytic flux plot"
./analyze.py -p p1 -q flux -a an
echo "$0: Generating p1 relative error plot"
./analyze.py -p p1 -q re
echo "$0: Generating p1 loss plot"
./analyze.py -p p1 -q loss

# Run scaling study and generate plots
echo "$0: Running scaling study"
./nnVsn.py p1scalstud --run
echo "$0: Analyze scaling study"
./nnVsn.py p1scalstud

# Run p2 and generate flux plot
echo "$0: Running p2"
./p2-half_slab.py
echo "$0: Generating p2 flux plot"
./analyze.py -p p2 -q flux
