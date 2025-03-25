# Hansen et al. (1981) Climate Model
Python implementation of the simple climate model from [Hansen et al. (1981)](https://www.science.org/doi/10.1126/science.213.4511.957)

## Prerequisites

Python package requirements are in  `requirements.txt`:
- Using `pip`: `pip install -r requirements.txt`

## Usage

Example data and notebooks can be found in the [examples](./examples/) folder, however, here is a simple example of using the model to project global average temperature until 2100 using different Shared Socioeconomic Pathways (SSPs).

```py
import pandas as pd
import matplotlib.pyplot as plt

# import module
from hansen_et_al_1981 import HansenEtAl1981

# load forcing data
historical_forcing_df = pd.read_csv('examples/data/forcing/historical.csv')
ssps_forcing_df = pd.read_csv('examples/data/forcing/ssps.csv')

# initialize the scaling factor for each forcing variable (column)
scaling_factor = { col:1.0 for col in historical_forcing_df.columns[1:-1] }

# convert YEAR column to datetime and set as index
historical_forcing_df['YEAR'] = historical_forcing_df['YEAR'].astype(int)
historical_forcing_df = historical_forcing_df.set_index('YEAR')

# extract SSP names and create a color map
ssp_names = sorted(list(ssps_forcing_df['SSP'].unique()))
ssp_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
ssp_color_map = dict(zip(ssp_names, ssp_colors))

# instantiate model
model = HansenEtAl1981()

plt.figure(figsize=(12, 6))
for i, ssp in enumerate(ssp_names):
    # select data for the current SSP
    ssp_forcing_df = ssps_forcing_df[ssps_forcing_df['SSP'] == ssp]
    
    # initialize the model
    model.set_model_params(Tm=0.0, Td=0.0)

    # run the model
    results_df = model.run(historical_forcing_df, ssp_forcings_df=ssp_forcing_df, scaling_factor=scaling_factor)
        
    # plot results
    plt.plot(results_df.loc[2000:].index, results_df.loc[2000:]['Pred Anom'], label=ssp, color=ssp_color_map[ssp])

plt.title('Global Temperature Pred Anomaly from Pre-Industrial Average (1850-1900)')
plt.xlabel('Year')
plt.ylabel('Temp Pred Anomaly (°C)')
plt.legend(framealpha=1)
plt.grid(which='major', linestyle='-', linewidth='0.5', color='gray')
plt.minorticks_on()
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
plt.show()
```
![](./examples/figures/temp_by_ssp.png)