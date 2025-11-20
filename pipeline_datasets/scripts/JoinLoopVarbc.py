import sys
import os
import getopt
import csv
import pandas as pd

YEAR = os.environ["YEAR"]
DOMAIN = os.environ["DOMAIN"]

# read the VARBC dataframe from -i
def get_varbc_df(file_path):
    varbc_df = pd.read_csv(file_path, index_col=False)
    return varbc_df

# read CCMA files from -c
def ccma_to_VARBC(filename):
    # Remove the filename extension
    filename_without_extension = filename.split('.')[0]
    # Take only the part after the 3rd underscore
    part_after_underscore = filename_without_extension.split('_')[3]
    # Add an underscore before the last 2 characters
    underscored_string = part_after_underscore[:-2] + '_' + part_after_underscore[-2:]
    # Append 4 zeros at the end
    transformed_filename = underscored_string + '0000'
    return transformed_filename

def calculate_statistics(subset_df, simple_df, column_prefix):
    data = subset_df[column_prefix]
    stats = {
        f"mean_{column_prefix}": round(data.mean(), 4),
        f"var_{column_prefix}": round(data.var(), 4),
        f"q25_{column_prefix}": round(data.quantile(q=0.25), 4),
        f"q50_{column_prefix}": round(data.quantile(q=0.5), 4),
        f"q75_{column_prefix}": round(data.quantile(q=0.75), 4)
    }
    
    return stats

def main():
    VARBC_FILE = os.environ.get("VARBC_OUTPUT_FILE") or "/perm/nld3863/pipeline_datasets/datasets/varbc_{DOMAIN}_common_sat_sen.csv"
    ODB_DIR = os.environ.get("MERGED_ODB_DIR") or "/perm/nld3863/pipeline_datasets/datasets/merged_{YEAR}_tables"

    varbc_df = get_varbc_df(VARBC_FILE)
    csv_files = [f for f in os.listdir(ODB_DIR) if f.endswith(".csv")]

    allDates_df = pd.DataFrame()
    for filename in csv_files:
        file_path = os.path.join(ODB_DIR, filename)
        if os.path.isfile(file_path):
            print(f"Filepath: {file_path}")
            df = pd.DataFrame(columns=['time', 'sat', 'sensor', 'channel'])
            large_df = pd.DataFrame()
            try:
                ccma_df = pd.read_csv(file_path, index_col = False)
                ccma_df.columns = ['date', 'time', 'lat', 'lon', 'sat', 'sensor', 'channel', 'fg_depar', 'an_depar', 'biascorr_fg', 'obs', 'peak_hPa', 'q25_anchors', 'q50_anchors', 'q75_anchors']
                time = ccma_to_VARBC(filename)
                print("time: ",time)
                ccma_df['time'] = time
                sat = ccma_df['sat'].unique()
                sen = ccma_df['sensor'].unique()
                chan = ccma_df['channel'].unique()

                print('sat:', sat)
                print('sen:', sen)

                for sa in sat:
                    for se in sen:
                        for ch in chan:
                            subset_df = ccma_df[(ccma_df['sat'] == sa) & (ccma_df['sensor'] == se) & (ccma_df['channel'] == ch)]
                            if not subset_df.empty:
                                simple_df = pd.DataFrame({
                                    'time': subset_df['time'].unique(),
                                    'sat': subset_df['sat'].unique(),
                                    'sensor': subset_df['sensor'].unique(),
                                    'channel': subset_df['channel'].unique()
                                    })
                                print("subset_df:", subset_df)
                                all_stats = {}
                                for column in ['fg_depar', 'an_depar', 'biascorr_fg', 'obs']:
                                    stats = calculate_statistics(subset_df=subset_df, simple_df=simple_df, column_prefix=column)
                                    all_stats.update(stats)
                                stats_df = pd.DataFrame([all_stats])

                                print("stats_df:", stats_df)
                                df = pd.concat([simple_df.reset_index(drop=True), stats_df], axis=1)
                                # add anchors and hPa:
                                anchor_cols = ['q25_anchors', 'q50_anchors', 'q75_anchors']
                                for col in anchor_cols:
                                    df[col] = subset_df[col].median()
                                print("df end iteration with anchors:", df)
                            else:
                                print("subset_df is empty")
                                
                            if large_df.empty:
                                large_df = df
                            else:
                                large_df = pd.concat([large_df, df])
                           
                            print("large df:", large_df.shape)
            except Exception as e:
                 print(f"An error occurred while processing file {file_path}: {e}")
                 continue

        if allDates_df.empty:
            allDates_df = large_df
        else:
            allDates_df = pd.concat([allDates_df, large_df])

    allDates_df = allDates_df.groupby(['time', 'sat', 'sensor', 'channel'], as_index=False).mean() # necessary to avoid duplicate rows with slightly different anchor values when 2 files have same datagroup at overlapping time

    print("varbc", varbc_df.head())
    merged_df = pd.merge(varbc_df, allDates_df, how='inner', on=['time', 'sat', 'sensor', 'channel'])
    merged_df = merged_df.groupby(['time', 'sat', 'sensor', 'channel', 'ndata','npred', 'pred_id'], as_index=False).mean()# make sure no duplicates are left
    merged_df['time_formatted'] = pd.to_datetime(merged_df['time'], format='%Y%m%d_%H%M%S')
    min_time = merged_df['time_formatted'].min()
    max_time = merged_df['time_formatted'].max()
    time_range = max_time - min_time

    merged_df['normalized_time'] = ((merged_df['time_formatted'] - min_time) / time_range).round(6)
    merged_df['cycle'] = merged_df['time_formatted'].dt.hour
    merged_df = merged_df.drop(['time_formatted'], axis=1)
    merged_df = merged_df.drop_duplicates()

    output_file = f"/perm/nld3863/pipeline_datasets/datasets/final_df/big_df_stats_{YEAR}.csv"
    merged_df.to_csv(output_file, index=False)

if __name__ == "__main__":
    main()
