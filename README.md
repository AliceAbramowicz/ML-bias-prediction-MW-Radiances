# A ML based approach for bias correction of microwave radiances in Regional NWP

## Collaborators: 

Alice Abramowicz, Isabel Monteiro, Kirien Whan, Irene Garcia Marti & Sanne Willems.

## Project description:

HARMONIE-AROME (H-A) is a regional Numerical Weather Prediction (NWP) model used for short-range weather forecasting in several operational centers across Europe. 
This NWP model relies on data assimilation (DA), which aims to define the most accurate conditions of the state of the atmosphere. 

Microwave radiance observations are crucial in NWP, but they are subject to systematic biases that need to be considered in the assimilation process. 
Currently, in H-A, bias correction of microwave radiance observations is performed using linear models for the bias in an adaptive approach using the variational assimilation system VarBC. 
Although this approach is effective in a global NWP, it comes with challenges for regional NWP.

This research explores an alternative way to correct for the biases of microwave radiances before they enter the DA process using machine learning (ML) techniques, such as neural networks (NN), adaptive boosting (AB), and random forests (RF). 
These ML bias predictions, found offline, are then integrated into a hybrid ML-DA which is used in an operational setup of H-A. 
Subsequently, the forecasts arising from the ML-DA approach are compared with the  forecasts arising from the traditional DA system in H-A. 
The development of a bias-correction prototype for microwave radiance observations offers significant potential for advancement in several critical areas. 
First, it could enable more precise bias corrections in high-resolution sub-kilometer domains, which are currently challenging for the traditional VarBC methods. 
Secondly, the ML approach could accelerate the spin-up phase for the next generation of satellites, enhancing the efficiency of microwave radiance assimilation for operational forecasting.

## Table of Contents:
1. Data engineering
    1. ODB preparation
    2. VarBC preparation
    3. Create full datasets
2. Datasets
3. Data exploration
    1. Data distributions
    2. Data correlations
4. ML models fitting
    1. Neural Networks (NN)
    2. Adaptive Boosting (AB)
    3. Random Forest (RF)

## Practical information: how to run the project

To execute the full data-engineering workflow, you will need access to your NWP system’s observational files:

- odb_ccma files
- VarBC.cycle files

If these are unavailable, you may skip the data-engineering pipeline and instead use the datasets provided in **Section 2 (Datasets)**.

## Data Engineering
For this project, we need to create a training and a testing dataset. 
To ensure generalizability, we use different geographical domains for training and testing: the 
- Training: the DINI domain
- testing: the Dutch domain.

<img width="549" height="547" alt="DINI_Dutch_Domains" src="https://github.com/user-attachments/assets/d01d25ef-ef4f-4ed6-9d1a-9e47ae7e137d" />

Configuration files are provided for both datasets:

- `config/dini.yaml`: training
- `config/dutch.yaml`: tesing

Adjust these files as needed, then generate datasets using the command line: 

```python3 pipeline_dataset.py --config config/CONFIG_FILE.yaml```

The only requirement is to already have the data from the `odb_ccma.tar` and `VARBC.cycle` files for you period(s) and domain(s) interest.

The pipeline will then take care for you of the tedious tasks of:



1. ODB preparation

    1. `untar_odb.sh`
    
        untar your odb_ccma files, and reorganize them in directories based on their _$yy$mm$dd$cy_ inside new directories in **ccma_output_dir**


    2. `WriteODBreq_sat.sh`

        From the CCMA files in **ccma_output_dir**, make an SQL request to fetch, for each satellite-sensor-channel, the variables:
        - first-guess departures
        - analysis departures
        - bias-corrected first guess
        - observation values

        Store your resulting satellite ODB tables in **sat_tables_dir** in csv files for each _$yy$mm$dd$cy_ .

    3. `WriteODBreq_anchors.sh`
        From the CCMA files in **ccma_output_dir**, make an SQL request to fetch, for anchor observations (radiosondes and aircrafts), the variables:
        - obseration values
        - vertical height

        Store the results in **anchors_dir** in csv files for each _$yy$mm$dd$cy_.
    
    4. `chan_peaks.py`

        In your satellite ODB tables in **sat_tables_dir** from the previous step, include the weight function peak height (hPa) of each channel. Save the new satellite ODB tables in the same directory **sat_tables_dir** with the suffix *_with_hPa.csv.

    5. `stats_radiosondes.py`

        Merge the anchor observations from radiosondes and aircraft to the satellite data at their corresponding hPa levels. Save the final ODB tables in **merged_odb_dir** and save them as merged_ALL_{domain}_{yy}{mm}{dd}{cy}.csv

2. VarBC preparation

    1. `makeDirectories_varbc.sh`

        take as inputs the VarBC.cycle files and reorganize them into directories based on $yy$mm$dd$cy inside a new folder named **dest_varbc_dir**. 

    2. `varbc_dataset_preparation.py`

        Use the reorganized VarBC cycles as inputs -i, create an output directory -o, and gather the VarBC cycles into a unified dataframe. Save it in a csv file **varbc_output_file**.

3. Create full datasets

    1. `JoinLoopVarbc.py`

        Read the VarBC dataframe from your input -i and the CCMA files from -c; 
        Compute statistics from:
        - first-guess departures 
        - analysis departures
        - bias-corrected first guess
        - observation values
        in the CCMA files;
        Match the rows from VarBC and CCMA files based on their ['time', 'sat', 'sensor', 'channel'];
        Create `big_df_stats_2021.csv` and `big_df_stats_2023.csv` and save them in **final_df**.

P.S. All files and directories indicated in **bold** in the above description can be defined in the config files.

Once you have run this pipeline for both your training and testing sets, you can create your final X and y sets for training and testing with `create_X_train_test.py`

From `big_df_stats_2021.csv`, `big_df_stats_2023.csv`:
- Fill in the missing values with 0 (i.e., for the covariances of non-existent predictors);
- Take care of categorical and numerical variables;
- Design new features;
- Scale variables;
- Create the final X_train, y_train, X_test, y_test (as well as their scaled counterparts).

## ML models fitting: results

After performing a hyperparameter search for the Random Forest (RF), Adaptive Boosting (AB) and Neural Network (NN), we obtain the following results for our models.

**RF (mse: 0.0199, r<sup>2</sup>: 0.8524):**

<img width="600" height="600" alt="RF_ytest_VS_ypred_colored_pred" src="https://github.com/user-attachments/assets/ea76f2c9-c9bb-4c90-bada-2ecbe3054f56" />

**AB (mse: 0.0253, r<sup>2</sup>: 0.8123):**

<img width="600" height="600" alt="AB_ytest_VS_ypred_colored_preds" src="https://github.com/user-attachments/assets/7cec7926-fcc4-4dae-8d94-3861ed90c03b" />

**NN (mse: 0.0269, r<sup>2</sup>: 0.8006):**

<img width="600" height="600" alt="NN_ytest_VS_ypred_colored_pred_exp" src="https://github.com/user-attachments/assets/b6aaf4f8-bd6d-471e-916c-59d7eeaaebdc" />

## Implementing ML bias correction in an NWP experiment (offline setting)

Once the ML inference is performed, feed your new ML bias coefficient predictions back to the `VARBC.cycles` of your experiment with the script `update_VarBC.py`. You can now arrange the scripts of your NWP to read these ML-based VarBC.cycle files during the analysis. In H-A (cycles 43-46-49), you can do so with the script `Fetch_assim_data`.

The workflow of the hybrid DA system implementing the ML bias correction is illustrated in the context of H-A in the followinf Figure:

<img width="2322" height="1411" alt="VarBC_ML_workflow" src="https://github.com/user-attachments/assets/21f578ec-a0e9-4840-b665-9421e916f30e" />

Make sure that VarBC does not activate to update your ML bias predictions. This can be done by imposing a very high stiffness parameter $M^j$ in the adaptivity matrix $B_\beta$ in the variational cost fuction: 

<img width="1409" height="89" alt="Variational_J_bias" src="https://github.com/user-attachments/assets/d52ac78a-5600-41f9-a6ae-f1534e7dec14" />


In H-A (cycles 43-46-49), you can do so in your NWP experiment namelist `harmonie_namelists.pm` by setting NBG very high for the sensors of interest:
```
varbc=(
 NAMVARBC=>{
 },
 NAMVARBC_RAD=>{
    ...
    'NBG_$SENSOR' ==> ...
 }
 ```
