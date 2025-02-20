#!/usr/bin/env python

"""climate_model.py: Module which contains the climate model implementation."""

__author__ = "Sean Kelley"
__license__ = "MIT"
__version__ = "1.0.0"

import math

import pandas as pd


# region Default hyperparameter values
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
# endregion

class HansenEtAl1981:
    '''
    Hansen et al. 1981 climate model implementation.
    '''
    def __init__(self, climate_sensitivity:float=None, diffusivity:float=None, mixed_layer_depth:float=None,
                       deep_ocean_depth:float=None, dt:float=None, Cm:float=None, Cd:float=None, ht:float=None, Tm:float=None, 
                       Td:float=None, alpha:float=None, a:float=None, b:float=None) -> None:
        '''
        Initialize the Hansen et al. 1981 climate model. If parameters aren't provided, the default values are used.

        :param float climate_sensitivity: Climate sensitivity
        :param float diffusivity: Diffusivity
        :param float mixed_layer_depth: Mixed layer depth
        :param float deep_ocean_depth: Deep ocean depth
        :param float dt: Time step in seconds
        :param float Cm: Mixed layer heat capacity
        :param float Cd: Deep ocean heat capacity
        :param float ht: Diffusivity*density*specific ht
        :param float Tm: Initial mixed layer temperature
        :param float Td: Initial deep ocean temperature
        :param float alpha: Thermal expansion coefficient
        :param float a: Semi-empirical constant, a
        :param float b: Semi-empirical constant, b
        '''
        self.set_model_params(climate_sensitivity, diffusivity, mixed_layer_depth,
                              deep_ocean_depth, dt, Cm, Cd, ht, Tm, Td, alpha, a, b)

        # pre industrial avg
        # TODO: maybe this should be calculated from the historical data
        self.pre_industrial = 0.07

    def set_model_params(self, climate_sensitivity:float=lmbd_DEFAULT,
                 diffusivity:float=K_DEFAULT,
                 mixed_layer_depth:float=dm_DEFAULT,
                 deep_ocean_depth:float=dd_DEFAULT,
                 dt:float=dt_DEFAULT,
                 Cm:float=Cm_DEFAULT,
                 Cd:float=Cd_DEFAULT,
                 ht:float=ht_DEFAULT,
                 Tm:float=Tm_DEFAULT,
                 Td:float=Td_DEFAULT,
                 alpha:float=alpha_DEFAULT,
                 a:float=a_DEFAULT,
                 b:float=b_DEFAULT) -> None:
        '''
        Set model parameters. This allows the user to re-run the model without reinstantiating the object.
        If parameters aren't provided, the default values are used.

        :param float climate_sensitivity: Climate sensitivity
        :param float diffusivity: Diffusivity
        :param float mixed_layer_depth: Mixed layer depth
        :param float deep_ocean_depth: Deep ocean depth
        :param float dt: Time step in seconds
        :param float Cm: Mixed layer heat capacity
        :param float Cd: Deep ocean heat capacity
        :param float ht: Diffusivity*density*specific ht
        :param float Tm: Initial mixed layer temperature
        :param float Td: Initial deep ocean temperature
        :param float alpha: Thermal expansion coefficient
        :param float a: Semi-empirical constant, a
        :param float b: Semi-empirical constant, b
        '''
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

    def run(self, historical_forcing_df: pd.DataFrame,
            ssps_forcing_df: pd.DataFrame = None,
            scaling_factor: dict[str, float] = None,
            return_forcings: bool = False) -> pd.DataFrame:
        '''
        Run the Hansen et al. 1981 climate model on the given forcing data.

        :param pd.DataFrame historical_forcing_df: Historical forcing data
        :param pd.DataFrame ssps_forcing_df: Forcing data for one SSP
        :param scaling_factor: Factor by which to scale each forcing factor
        :type scaling_factor: dict[str, float] or None
        :param return_forcings: Whether to return the forcing values in the output DataFrame
        :type return_forcings: bool or None
        :return: DataFrame containing the results of the model run: deep ocean temperature, mixed layer temperature, and global temperature anomaly
        :rtype: pd.DataFrame
        '''
        # TODO: take in only one dataframe assuming user has already concatenated the historical and SSPs forcing data
        # concatenate historical and SSPs forcing data
        if ssps_forcing_df is not None:
            ssps_forcing_df = ssps_forcing_df.set_index('YEAR')
            forcing_df = pd.concat([historical_forcing_df, ssps_forcing_df])
        else:
            forcing_df = historical_forcing_df

        # get forcing factors with available scaling factor
        forcing_factors = list(
            set(scaling_factor.keys()).intersection(set(forcing_df.columns)))

        # TODO: add error handling if no forcing factors are found

        # create an empty DataFrame to store the results
        columns = ['Forcing', 'Deep dT', 'Mixed dT', 'Pred Anom']
        if return_forcings:
            columns = forcing_factors+columns
        results_df = pd.DataFrame(columns=columns, index=forcing_df.index)

        # iterate over each row in the forcing DataFrame
        for year, row in forcing_df.iterrows():
            # for each scaling factor, multiply by the forcing value, ignoring NaN forcing values
            forcing_vals = [(row[factor] if not math.isnan(
                row[factor]) else 0)*scaling_factor[factor] for factor in forcing_factors]

            F = sum(forcing_vals)

            # update the deep ocean temperature
            # =C2 + $O$13*$O$16*(D2-C2)/(0.5*$O$15*($S$7+$S$8))
            self.Td += self.dt*self.ht * \
                (self.Tm-self.Td)/(0.5*self.Cd*(self.dm+self.dd))

            # update the mixed layer temperature
            # =D2+ $O$13*( B3- $O$16*(D2-C2)/(0.5*($S$7+$S$8))-(D2/$O$6))/$O$14
            self.Tm += self.dt*(F-self.ht*(self.Tm-self.Td) /
                                (0.5*(self.dm+self.dd))-(self.Tm/self.l))/self.Cm

            # calculate the global temperature anomaly
            anom = self.Tm - self.pre_industrial

            # append the results to the DataFrame
            results_row = [F, self.Td, self.Tm, anom]
            if return_forcings:
                results_row = forcing_vals+results_row
            results_df.loc[year] = results_row

        return results_df.infer_objects()
