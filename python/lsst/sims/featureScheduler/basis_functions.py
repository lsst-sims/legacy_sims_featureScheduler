import numpy as np
import features
import utils


class Base_basis_function(object):
    """
    Class that takes features and computes a reward fucntion
    """

    def __init__(self, survey_features=None, condition_features=None):
        """

        """
        if survey_features is None:
            self.survey_features = []
        else:
            self.survey_features = survey_features
        if condition_features is None:
            self.condition_features = []
        else:
            self.condition_features = condition_features

    def add_observation(self, observation):
        for feature in self.survey_features:
            feature.add_observation(observation)

    def update_conditions(self, conditions):
        for feature in self.condition_features:
            feature.update_conditions(conditions)

    def __call__(self):
        """
        Return a reward healpix map or a reward scalar.
        """
        pass


class Target_map_basis_function(Base_basis_function):
    """
        
    """
    def __init__(self, filtername='r'):
        self.survey_features = [features.N_observations(fltername=filtername)]
        self.survey_features.append(features.N_obs_reference())
        
