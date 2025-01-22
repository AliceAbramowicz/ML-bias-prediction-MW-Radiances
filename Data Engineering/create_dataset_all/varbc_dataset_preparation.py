import sys
import os
import getopt
import pandas as pd
import numpy as np

def usage():
    print("Options:")
    print("  -i : VARBCINP directory")
    print("  -o : VARBCOUT directory")
    print("  -S : SATLIST")
    print("  -s : SENLIST")
    print("  -h : Help")

def readVARBC(file_path):
    result=[]
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for i in range(len(lines)):
            if lines[i].startswith('ndata'):
                ndata_value = int(lines[i].split('=')[1].strip())
                if ndata_value > 1:
                    ndata = int(lines[i].split('=')[1].strip())
                    
                    key = lines[i - 2].split('=')[1].strip()
                    npred = int(lines[i + 1].split('=')[1].strip())
                    predcs = lines[i + 2].split('=')[1].strip()
                    param0 = lines[i + 3].split('=')[1].strip()
                    params = lines[i + 4].split('=')[1].strip()
                    predmean = lines[i + 7].split('=')[1].strip()
                    predxcnt = lines[i + 6].split('=')[1].strip()
                    predxcov = lines[i + 8].split('=')[1].strip()
                    result.append((ndata, key, npred, predcs, param0, params, predmean, predxcov))
        return(result)
 
def main():
    VARBCINP = "DUMMY"
    VARBCOUT = "DUMMY"
    SATLIST = "5"
    SENLIST = "3"

    try:
        opts, args = getopt.getopt(sys.argv[1:], "i:o:S:s:h")
    except getopt.GetoptError as err:
        print(str(err))
        usage()
        sys.exit(1)

    for opt, arg in opts:
        if opt == '-h':
            usage()
            sys.exit(0)
        elif opt == '-i':
            VARBCINP = arg
        elif opt == '-o':
            VARBCOUT = arg
        elif opt == '-S':
            SATLIST = arg
        elif opt == '-s':
            SENLIST = arg

    if VARBCINP == "DUMMY":
        print("No VARBCINP directory provided")
        print("Try '{} -h' for more information".format(sys.argv[0]))
        sys.exit(1)

    if not os.path.exists(VARBCOUT):
        os.makedirs(VARBCOUT)
    else:
        print("Directory {} already exists. Please choose another name".format(VARBCOUT))
        sys.exit(1)

    print("Starting to loop over files ...")
    columns = ['time', 'sat', 'sensor', 'channel', 'ndata', 'npred', 'pred_id', 'param0', 'param', 'predmean']
    varbc_df = pd.DataFrame(columns=columns)

    # Go through all files within directory VARBCINP
    for root, dirs, files in os.walk(VARBCINP):
        for file in files:
            if file == "VARBC.cycle":
                # join root directory "root" to the file called VARBC.cycle
                file_path = os.path.join(root, file)
                YYYYMMDD = None
                HHMMSS_S = None
                with open(file_path, 'r') as f:
                    for i, line in enumerate(f):
                        # 2nd line to get time
                        if i == 1:
                            # store 2nd token in YYYYMMDD and 3rd in HHMMSS_S
                            YYYYMMDD, HHMMSS_S = line.split()[1:3]
                            break
                # If there's no date or cycle, skip this file
                if YYYYMMDD is None or HHMMSS_S is None:
                    continue
                # transform time string to 6 int with leading 0's if necessary
                HHMMSS = "{:06d}".format(int(HHMMSS_S))
                
                result = readVARBC(file_path)
                if len(result) == 0:
                    continue
                for tup in result:
                    ndata, key, npred, predcs, param0, params, predmean, predxcov = tup
                    name_key = '_'.join(key.split())
                    print(" predictors:", predcs)
                    sat, sensor, channel = key.split()
                    time = "{}_{}".format(YYYYMMDD, HHMMSS)
                    #file_out = f"VARBC_{name_key}_fmt.csv"
                    print(time)
                    for SAT in SATLIST.split(':'):
                        for SEN in SENLIST.split(':'):
                            if ((sensor == SEN) and (sat == SAT)):
                                predxcov_values = [float(value) for value in predxcov.split()]
                                predxcov_matrix = np.array(predxcov_values).reshape(npred, npred)
                                for n in range(npred):
                                    varbc_df=varbc_df.append(pd.Series({
                                        'time': time,
                                        'sat': sat,
                                        'sensor': sensor,
                                        'channel': channel,
                                        'ndata': ndata,
                                        'npred': npred,
                                        'pred_id': int(predcs.split()[n]),
                                        'param0': float(param0.split()[n]),
                                        'param': float(params.split()[n]),
                                        'predmean': float(predmean.split()[n]),
                                        #dictionary comprehension containing column names & covariance values
                                        **{'predxcov_'+str(i+1): predxcov_matrix[i, n] for i in range(npred)}}), ignore_index=True)

    varbc_df.to_csv(os.path.join(VARBCOUT, "varbc_dataset.csv"), header=True, index=False)
    print(f"Output available in {VARBCOUT}")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("No command line arguments provided")
        print("Try '{} -h' for more information".format(sys.argv[0]))
        sys.exit(1)
    main()
