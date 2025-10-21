from enum import IntEnum

class MetricType(IntEnum):
    INNER_PRODUCT = 0  #< maximum inner product search
    L2 = 1             #< squared L2 search
    L1 = 2             #< L1 (aka cityblock)
    Linf = 3           #< infinity distance
    Lp = 4             #< L_p distance, p is given by a faiss::Index metric_arg
    # some additional metrics defined in scipy.spatial.distance
    Canberra = 20
    BrayCurtis = 21
    JensenShannon = 22
    # sum_i(min(a_i, b_i)) / sum_i(max(a_i, b_i)) where a_i, b_i > 0
    Jaccard = 23
    # Squared Eucliden distance, ignoring NaNs
    NaNEuclidean = 24
    # Gower's distance - numeric dimensions are in [0,1] and categorical dimensions are negative integers
    GOWER = 25