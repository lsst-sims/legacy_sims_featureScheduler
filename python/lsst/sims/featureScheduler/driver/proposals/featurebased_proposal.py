from lsst.ts.scheduler.proposals import AreaDistributionProposal
from lsst.ts.scheduler.setup import EXTENSIVE

__all__ = ["FeatureBasedProposal"]

class FeatureBasedProposal(AreaDistributionProposal):

    def __init__(self, propid, name, confdict, skymodel):

        AreaDistributionProposal.__init__(self, propid, name, confdict, skymodel)

    def register_observation(self, observation):

        self.last_observation = observation.get_copy()
        self.last_observation_was_for_this_proposal = False

        if self.propid not in observation.propid_list and not self.params.accept_serendipity:
            return None

        fieldid = observation.fieldid
        filter = observation.filter

        tfound = None
        for target in self.winners_list:
            if self.observation_fulfills_target(observation, target):
                tfound = target
                break
        if tfound is None:
            for target in self.losers_list:
                if self.observation_fulfills_target(observation, target):
                    tfound = target
                    break

        if tfound is not None:
            self.log.log(EXTENSIVE, "register_observation: %s" % (target))

            target.targetid = observation.targetid
            target = self.survey_targets_dict[fieldid][filter]
            target.visits += 1
            target.progress = float(target.visits) / target.goal
            target.last_visit_time = observation.time


            self.survey_targets_visits += 1
            if self.survey_targets_goal > 0:
                self.survey_targets_progress = float(self.survey_targets_visits) / self.survey_targets_goal
            else:
                self.survey_targets_progress = 0.0
            self.survey_filters_visits_dict[filter] += 1
            if self.survey_filters_goal_dict[filter] > 0:
                self.survey_filters_progress_dict[filter] = \
                    float(self.survey_filters_visits_dict[filter]) / self.survey_filters_goal_dict[filter]
            else:
                self.survey_filters_progress_dict[filter] = 0.0

            self.last_observation_was_for_this_proposal = True

        return tfound



