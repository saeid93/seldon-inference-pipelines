import logging
import os
import numpy as np
logger = logging.getLogger(__name__)


class EnsembleCombiner:
    def aggregate(self, features, names=[], meta=[]):
        final_output = {}
        for feature in features:
            model_name = feature['model_name']
            model_percentages = feature['percentages']
            model_ouptut = {}
            max_prob_percentage = max(model_percentages)
            max_prob_class = np.argmax(model_percentages)
            model_ouptut = {
                'max_prob_class': int(max_prob_class),
                'max_prob_percentage': max_prob_percentage,
                'percentages': model_percentages
            }
            final_output[model_name] = model_ouptut
            logger.info("-" * 80)
            logger.info(f"model_name: {model_name}")
            logger.info(f"max_prob_class: {max_prob_class}")
        ensemble_max_prob_percentages = sum(
            map(lambda x: np.array(
                x['percentages']),
                final_output.values()))/len(final_output)
        ensemble_max_prob_percentage = max(
            ensemble_max_prob_percentages)
        ensemble_max_prob_class = np.argmax(
            ensemble_max_prob_percentages)
        ensemble_output = {
            'max_prob_class': int(ensemble_max_prob_class),
            'max_prob_percentage': float(ensemble_max_prob_percentage),
            'percentages': ensemble_max_prob_percentages.tolist() 
        }
        final_output['ensemble_output'] = ensemble_output
        return final_output
