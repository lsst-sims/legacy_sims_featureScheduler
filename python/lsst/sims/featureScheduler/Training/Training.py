import numpy as np
import lsst.sims.featureScheduler.Training as opt
from lsst.sims.featureScheduler.observatory import Speed_observatory




class BlackTraining(object):
    def __init__(self, preferences, gray_train = False, custom_period = 1):

        self.pref       = preferences

        self.survey_length = 2  # days
        self.surveys = []
        # Define what we want the final visit ratio map to look like
        survey_filters = ['r']
        for f in survey_filters:
            self.bfs = []
            self.bfs.append(opt.Slewtime_basis_function_cost(filtername=f))
            self.bfs.append(opt.Visit_repeat_basis_function_cost(filtername=f,survey_filters=survey_filters))
            self.bfs.append(opt.Target_map_basis_function_cost(filtername=f, survey_filters=survey_filters))
            self.bfs.append(opt.Normalized_alt_basis_function_cost(filtername=f))
            self.bfs.append(opt.Hour_angle_basis_function_cost())
            self.bfs.append(opt.Depth_percentile_basis_function_cost())
            weights = np.array([5,2,1,1,2,1])
            self.surveys.append(opt.Simple_greedy_survey_fields_cost(self.bfs, weights, filtername=f, block_size= 10))

    def DE_opt(self, N_p, F, Cr, maxIter, D, domain, gray_trianing = False):
        self.D               = D
        self.domain          = domain
        self.optimizer       = fs.DE_optimizer(self, N_p, F, Cr, maxIter, gray_training = gray_trianing)

    def target(self, x):
        x[0] = 5 # reduce redundant solutions
        for survey in self.surveys:
            survey.basis_weights = x
        scheduler = opt.Core_scheduler_cost(self.surveys)
        observatory = Speed_observatory()
        observatory, scheduler, observations = opt.sim_runner(observatory, scheduler,
                                                         survey_length=self.survey_length)
        return -1 * opt.simple_performance_measure(observations, self.pref)

    def refined_individual(self):
        return np.zeros(self.D)




N_p     = 50      # number of candidate solutions that are supposed to explore the space of solution in each iteration, rule of thumb: ~10*D
F       = 0.8    # algorithm meta parameter (mutation factor that determines the amount of change for the derivation of candidate solutions of the next iteration)
Cr      = 0.8    # algorithm meta parameter (crossover rate)
maxIter = 100     # maximum number of iterations. maximum number of function evaluations = N_p * maxIter,
Domain  = np.array([[0,10], [0,10], [0,10], [0,10], [0,10], [0,10]]) # Final solution would lie in this domain
D       = 6      # weights dimension

preferences     = [1,1]  # to define the objective function based on scientific preferences, can have any dimension
#P1: slew time     * -1
#P2: N_observation * 1


train   = BlackTraining(preferences)
train.DE_opt(N_p, F, Cr, maxIter, D, Domain)






