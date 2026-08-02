import numpy as np
from torchpack.utils.config import configs


def make_new_query(query, top_k, rerank_type):
    new_a_g = [query['g_desc']]
    for i in range(len(top_k)):
        k_data = top_k[i]
        if rerank_type == 'avgQE':
            new_a_g.append(k_data['g_desc'])
        else:
            new_a_g.append(k_data['g_desc'] * np.linalg.norm(query['g_desc'] - k_data['g_desc']) ** configs.rerank.alpha)
    new_a_g = np.vstack(new_a_g)
    new_a_g = np.average(new_a_g, axis=0)
    return new_a_g
