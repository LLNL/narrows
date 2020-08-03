# genplots.sh

# Run full_slab and generate plots
echo "$0: Running full_slab"
../bin/narrows full_slab.yaml
echo "$0: Generating full_slab analytic flux plot"
./analytic_full_slab.py
echo "$0: Generating full_slab relative error plot"
./analyze.py full_slab -q re
echo "$0: Generating full_slab loss plot"
./analyze.py full_slab -q loss

# Run scaling study and generate plots
echo "$0: Running scaling study"
./nnVsn.py --run
echo "$0: Analyze scaling study"
./nnVsn.py

# Run half_slab and generate flux plot
echo "$0: Running half_slab"
../bin/narrows half_slab.yaml
echo "$0: Generating half_slab flux plot"
./analyze.py half_slab -q flux
