import numpy as np
import lsst.sims.featureScheduler as fs
from speed_observatory import Speed_observatory
import os



class BlackTraining():
    def __init__(self, preferences, gray_train = False, custom_period = 1):

        self.pref       = preferences

        self.survey_length = 1  # days
        # Define what we want the final visit ratio map to look like
        target_map = fs.standard_goals()['r']
        self.bfs = []
        self.bfs.append(fs.Depth_percentile_basis_function())
        self.bfs.append(fs.Target_map_basis_function(target_map=target_map))

    def DE_opt(self, N_p, F, Cr, maxIter, D, domain, gray_trianing = False):
        self.D               = D
        self.domain          = domain
        self.optimizer       = fs.DE_optimizer(self, N_p, F, Cr, maxIter, gray_training = gray_trianing)

    def target(self, x):
        weights = x
        survey = fs.Simple_greedy_survey(self.bfs, weights)
        scheduler = fs.Core_scheduler([survey])
        observatory = Speed_observatory()
        observatory, scheduler, observations = fs.sim_runner(observatory, scheduler,
                                                         survey_length=self.survey_length)
        return -1 * fs.simple_performance_measure(observations, self.pref)

    def refined_individual(self):
        return np.array([0,0])




N_p     = 5      # number of candidate solutions that are supposed to explore the space of solution in each iteration, rule of thumb: ~10*D
F       = 0.8    # algorithm meta parameter (mutation factor that determines the amount of change for the derivation of candidate solutions of the next iteration)
Cr      = 0.8    # algorithm meta parameter (crossover rate)
maxIter = 10     # maximum number of iterations. maximum number of function evaluations = N_p * maxIter,
Domain  = np.array([[0,20], [0,20]]) # Final solution would lie in this domain
D       = 2      # weights dimension

preferences     = [1,1]  # to define the objective function based on scientific preferences, can have any dimension
#P1: slew time     * -1
#P2: N_observation * 1


train   = BlackTraining(preferences)
train.DE_opt(N_p, F, Cr, maxIter, D, Domain)






