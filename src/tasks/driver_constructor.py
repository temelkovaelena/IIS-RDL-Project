"""Custom task: recommend a constructor for a driver for the next season.

RelBench ships three entity tasks for rel-f1; this is a fourth, of the recommendation
kind (MAP@k).

Label: the set of constructor_id a driver has results with in the following season.
Split: by season, reusing task.val_timestamp / test_timestamp so the numbers stay comparable.

Careful: most drivers stay with the same constructor, so a "persistence" baseline is strong.
MAP is therefore reported three times: all / movers (drivers who switched) / rookies (no history).
"""
