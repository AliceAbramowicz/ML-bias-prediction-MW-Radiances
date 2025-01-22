import csv

"""
Keep only satellites and sensors which are in both the Dutch and the Dini domain
Remove extra covariance terms, considering that we never have more than 6 predictors for the remaining data groups
"""

def remove_rows(input_file, output_file):
    with open(input_file, 'r', newline='') as infile:
        reader = csv.DictReader(infile)
        columns_to_remove = ['predxcov_8', 'predxcov_7', 'predxcov_9', 'predxcov_10', 'time_formatted']
        header = [col for col in reader.fieldnames if col not in columns_to_remove]
        rows_to_keep = [{col: row[col] for col in header} for row in reader if row['sat'] not in ['4', '206', '224', '225'] and row['sensor'] not in  ['19', '16', '4']]

    with open(output_file, 'w', newline='') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows_to_keep)

input_file = '/hpcperm/nld3863/VarBC_dataset_2021/varbc_dataset_dutch_all_sat_sen.csv'
output_file = '/hpcperm/nld3863/VarBC_dataset_2021/varbc_dataset_dutch_common_sat_sen.csv'

remove_rows(input_file, output_file)
