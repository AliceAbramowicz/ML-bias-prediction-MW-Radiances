import os
import csv

directory = "/hpcperm/nld3863/loop_DUTCH_ALL"
#directory = "/hpcperm/nld3863/loop_DINI_ALL"

def convert_to_csv(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
        reader = csv.reader(infile, delimiter=' ')
        writer = csv.writer(outfile, delimiter=',')
        for row in reader:
            # Remove any empty strings caused by multiple spaces
            cleaned_row = [item for item in row if item.strip()]
            writer.writerow(cleaned_row)

# Keep only satellites and sensors which are in both the Dutch and the Dini domain
def remove_rows(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:    
        reader = csv.DictReader(infile)
        header = reader.fieldnames
        print(header)
        writer = csv.DictWriter(outfile, fieldnames=header)
        writer.writeheader()
        if header is None:
            print(f"Warning: No data found in {input_file}")
            return
        rows_to_keep = [row for row in reader if row['satellite_identifier@sat'] not in ['4', '206', '224', '225'] and row['sensor@hdr'] not in ['19', '16', '4']]
        writer.writerows(rows_to_keep)

for filename in os.listdir(directory):
    input_file = os.path.join(directory, filename)
    if (filename.endswith('.txt') and os.stat(input_file).st_size > 0):
        print(input_file)
        csv_file = os.path.join(directory, filename.replace('.txt', '.csv'))
        convert_to_csv(input_file, csv_file)
        output_file = os.path.join(directory, filename[:-4] + '_common.csv')
        remove_rows(csv_file, output_file)

