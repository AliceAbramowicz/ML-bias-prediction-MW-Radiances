import sys
import os
import getopt
import csv
import pandas as pd

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
        f"mean_{column_prefix}": round(data.mean(), 6),
        f"var_{column_prefix}": round(data.var(), 6),
        f"min_{column_prefix}": round(data.min(), 6),
        f"max_{column_prefix}": round(data.max(), 6),
        f"quantile_5_{column_prefix}": round(data.quantile(q=0.05), 6),
        f"quantile_25_{column_prefix}": round(data.quantile(q=0.25), 6),
        f"median_{column_prefix}": round(data.quantile(q=0.5), 6),
        f"quantile_75_{column_prefix}": round(data.quantile(q=0.75), 6),
        f"quantile_95_{column_prefix}": round(data.quantile(q=0.95), 6)
    }
    
    return stats

def main():
    try:
        opts, args = getopt.getopt(sys.argv[1:], "i:c:")
    except getopt.GetoptError as err:
        print(str(err))
        usage()
        sys.exit(1)

    for opt, arg in opts:
        if opt == '-i':
            VARBCINP = arg
        elif opt == '-c':
            CCMA = arg

    varbc_df = get_varbc_df(VARBCINP)
    csv_files = [file for file in os.listdir(CCMA) if file.endswith('common.csv')]

    allDates_df = pd.DataFrame()
    for filename in csv_files:
        file_path = os.path.join(CCMA, filename)
        if os.path.isfile(file_path):
            print(f"Filepath: {file_path}")
            df = pd.DataFrame(columns=['time', 'sat', 'sensor', 'channel'])
            large_df = pd.DataFrame()
            try:
                ccma_df = pd.read_csv(file_path, index_col = False)
                # DINI:
                #ccma_df.columns = ['date', 'time', 'lat', 'lon', 'sat', 'sensor', 'channel', 'update', 'fg_depar', 'an_depar', 'biascorr_fg', 'obsvalue', 'biascorr']
                # DUTCH:
                ccma_df.columns = ['date', 'time', 'lat', 'lon', 'sat', 'sensor', 'channel', 'update', 'fg_depar', 'an_depar', 'biascorr_fg', 'obsvalue']
                time = ccma_to_VARBC(filename)
                print("time: ",time)
                ccma_df['time'] = time
                sat = ccma_df['sat'].unique()
                sen = ccma_df['sensor'].unique()
                chan = ccma_df['channel'].unique()

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
                                for column in ['fg_depar', 'an_depar', 'biascorr_fg', 'obsvalue']:
                                    stats = calculate_statistics(subset_df=subset_df, simple_df=simple_df, column_prefix=column)
                                    all_stats.update(stats)
                                stats_df = pd.DataFrame([all_stats])
                                print("stats_df:", stats_df)
                                df = pd.concat([simple_df.reset_index(drop=True), stats_df], axis=1)

                                print("df end iteration:", df)
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
    
    print("varbc", varbc_df.head())
    merged_df = pd.merge(varbc_df, allDates_df, how='inner', on=['time', 'sat', 'sensor', 'channel'])
    merged_df['time_formatted'] = pd.to_datetime(merged_df['time'], format='%Y%m%d_%H%M%S')
    min_time = merged_df['time_formatted'].min()
    max_time = merged_df['time_formatted'].max()
    time_range = max_time - min_time

    merged_df['normalized_time'] = (merged_df['time_formatted'] - min_time) / time_range
    merged_df['normalized_time'] = merged_df['normalized_time'].round(6)
    merged_df['cycle'] = merged_df['time_formatted'].dt.hour
    merged_df = merged_df.drop(['time_formatted'], axis=1)
    merged_df = merged_df.drop_duplicates()
    merged_df.to_csv("big_df_stats_2021.csv", index=False)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("No command line arguments provided")
        print("Try '{} -h' for more information".format(sys.argv[0]))
        sys.exit(1)
    main()
