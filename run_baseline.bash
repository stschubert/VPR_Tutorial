source ~/.bashrc
conda activate vpr
for DIR in $(ls images/HTT_example)
do
    python3 hubble_baseline_demo.py --dataset $DIR
done
