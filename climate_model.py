#!/usr/bin/env python

"""climate_model.py: Module which contains the climate model implementation."""

__author__ = "Sean Kelley"
__license__ = "MIT"
__version__ = "1.0.0"

import math

import pandas as pd


#region Default hyperparameter values
# climate sensitivity
lmbd_DEFAULT = 0.80
# diffusivity
K_DEFAULT = 0.00010
# mixed layer depth
dm_DEFAULT = 100.0
# deep ocean depth
dd_DEFAULT = 900.0
# time step in seconds
dt_DEFAULT = 31536000.0
# mixed layer heat capacity 
Cm_DEFAULT = 421800000.0
# deep ocean heat capacity Cd
Cd_DEFAULT = 3796200000.0
# Diffusivity*density*specific ht
ht_DEFAULT = 421.8
# init mixed layer temp
Tm_DEFAULT = 0.0
# init deep ocean temp
Td_DEFAULT = 0.0
# thermal expansion coefficient
alpha_DEFAULT = 0.0002
# Semi-empirical constant, a
a_DEFAULT = 0.003
# Semi-empirical constant, b
b_DEFAULT = 0.049
#endregion

# TODO: does this need to be a class
class HansenEtAl1981:
    def __init__(self, climate_sensitivity=lmbd_DEFAULT, 
                 diffusivity=K_DEFAULT, 
                 mixed_layer_depth=dm_DEFAULT, 
                 deep_ocean_depth=dd_DEFAULT,
                 dt=dt_DEFAULT,
                 Cm=Cm_DEFAULT,
                 Cd=Cd_DEFAULT,
                 ht=ht_DEFAULT,
                 Tm=Tm_DEFAULT,
                 Td=Td_DEFAULT,
                 alpha=alpha_DEFAULT,
                 a=a_DEFAULT,
                 b=b_DEFAULT):
        
        self.l = climate_sensitivity
        self.K = diffusivity
        self.dm = mixed_layer_depth
        self.dd = deep_ocean_depth
        self.dt = dt
        self.Cm = Cm
        self.Cd = Cd
        self.ht = ht
        self.Tm = Tm
        self.Td = Td
        self.alpha = alpha
        self.a = a
        self.b = b

        # pre industrial avg
        # TODO: maybe this should be calculated from the historical data
        self.pre_industrial = 0.07

    def init_ocean_temp(self, Tm=0.0, Td=0.0):
        self.Tm = Tm
        self.Td = Td

    def run(self, historical_forcing_df:pd.DataFrame, ssps_forcing_df:pd.DataFrame=None, scaling_factor:dict=None, return_forcings:bool=False):
        '''
        Convert string representation of a number into a float.

            Parameters:
                x (str): specific string representation of a number
                scaling_factor (int): factor of 10 to scale up (positive) or down (negative)

            Returns:
                number (float): float value of string representation
        '''
        # TODO: take in only one dataframe assuming user has already concatenated the historical and SSPs forcing data
        # concatenate historical and SSPs forcing data
        if ssps_forcing_df is not None:
            ssps_forcing_df = ssps_forcing_df.set_index('YEAR')
            forcing_df = pd.concat([historical_forcing_df, ssps_forcing_df])
        else:
            forcing_df = historical_forcing_df

        # get forcing factors with available scaling factor
        forcing_factors = list(set(scaling_factor.keys()).intersection(set(forcing_df.columns)))

        # TODO: add error handling if no forcing factors are found

        # create an empty DataFrame to store the results
        columns = ['Forcing', 'Deep dT', 'Mixed dT', 'Pred Anom']
        if return_forcings:
            columns = forcing_factors+columns
        results_df = pd.DataFrame(columns=columns, index=forcing_df.index)

        # iterate over each row in the forcing DataFrame
        for year, row in forcing_df.iterrows():
            # for each scaling factor, multiply by the forcing value, ignoring NaN forcing values
            forcing_vals = [(row[factor] if not math.isnan(row[factor]) else 0)*scaling_factor[factor] for factor in forcing_factors]

            F = sum(forcing_vals)

            # update the deep ocean temperature
            # =C2 + $O$13*$O$16*(D2-C2)/(0.5*$O$15*($S$7+$S$8))
            self.Td += self.dt*self.ht*(self.Tm-self.Td)/(0.5*self.Cd*(self.dm+self.dd))

            # update the mixed layer temperature
            # =D2+ $O$13*( B3- $O$16*(D2-C2)/(0.5*($S$7+$S$8))-(D2/$O$6))/$O$14
            self.Tm += self.dt*(F-self.ht*(self.Tm-self.Td)/(0.5*(self.dm+self.dd))-(self.Tm/self.l))/self.Cm

            # calculate the global temperature anomaly
            anom = self.Tm - self.pre_industrial

            # append the results to the DataFrame
            results_row = [F, self.Td, self.Tm, anom]
            if return_forcings:
                results_row = forcing_vals+results_row
            results_df.loc[year] = results_row
        
        return results_df.infer_objects()