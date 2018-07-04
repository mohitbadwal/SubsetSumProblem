1:Extract new_data.zip


Run the scripts in the following order:

1: Subset_Generator.py(new_data.csv is read from this file)
2: feature(delay,variance_bucket).py
3: feature(LMH).py
4: feature(quarter_level).py
5: roll_up_to_subset.py
6:Payment_smh_feature.py
7: train_test(70_30).py
8: subset_model