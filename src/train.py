"""Training. Must be callable from the command line:

    python src/train.py --task driver-top3 --seed 0 --out output/rows.csv

Neighbour sampling is time-limited: every neighbour must be older than the seed node.
That is asserted, not assumed.
"""
