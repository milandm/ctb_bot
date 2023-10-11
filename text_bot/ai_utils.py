import logging
import os
from sklearn.metrics.pairwise import paired_cosine_distances, paired_euclidean_distances, paired_manhattan_distances
from scipy.stats import pearsonr, spearmanr
import numpy as np
from typing import List

import logging
import os
from sklearn.metrics.pairwise import paired_cosine_distances, paired_euclidean_distances, paired_manhattan_distances
from scipy.stats import pearsonr, spearmanr
import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer, util



def get_distance_scores(embeddings1, embeddings2):
    cosine_scores = 1 - (paired_cosine_distances([embeddings1], [embeddings2]))
    manhattan_distances = paired_manhattan_distances([embeddings1], [embeddings2])
    euclidean_distances = paired_euclidean_distances([embeddings1], [embeddings2])
    dot_products = [np.dot(emb1, emb2) for emb1, emb2 in zip([embeddings1], [embeddings2])]
    cos_sim_score_transformer = util.pytorch_cos_sim(embeddings1, embeddings2)

    distance_scores = {"cosine_scores": cosine_scores[0],
                       "manhattan_distances": manhattan_distances[0],
                       "euclidean_distances": euclidean_distances[0],
                       "dot_products": dot_products[0],
                       "cos_sim_score_transformer": cos_sim_score_transformer[0]}

    return distance_scores


def check_distance_scores_out(distance_scores):
    COSINE_SCORES_MAX_OUT = 0.6
    COSINE_SCORES_MIN_IN = 0.6
    DOT_PRODUCTS_MIN_IN = 0.7

    MANHATTAN_DISTANCES_MAX_IN = 15
    EUCLIDEAN_DISTANCES_MAX_IN = 0.75

    return distance_scores["cosine_scores"] < COSINE_SCORES_MAX_OUT \
        and distance_scores["dot_products"] < DOT_PRODUCTS_MIN_IN \
        and distance_scores["manhattan_distances"] > MANHATTAN_DISTANCES_MAX_IN \
        and distance_scores["euclidean_distances"] > EUCLIDEAN_DISTANCES_MAX_IN


@staticmethod
def is_close(a, b, threshold):
    return abs(a - b) <= threshold