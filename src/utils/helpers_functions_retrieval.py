import numpy as np
import torch

# def normalize_features(features):
#     """
#     Normalizes the features to unit length.
    
#     Args:
#         features (torch.Tensor): A 2D tensor of features.

#     Returns:
#         torch.Tensor: The normalized features.
#     """
#     return features / features.norm(dim=1, keepdim=True)



# def average_precision(retrieved, relevant):
#     """
#     Calculate Average Precision for a single query.
#     :param retrieved: List of retrieved item indices.
#     :param relevant: List of relevant item indices.
#     :return: Average Precision score.
#     """
#     retrieved = np.array(retrieved)
#     relevant = np.array(relevant)
#     rel_mask = np.in1d(retrieved, relevant)

#     cum_rel = np.cumsum(rel_mask)
#     precision_at_k = cum_rel / (np.arange(len(retrieved)) + 1)
#     average_precision = np.sum(precision_at_k * rel_mask) / len(relevant)
    
#     return average_precision

# def mean_average_precision(retrieved_lists, relevant_lists):
#     """
#     Calculate Mean Average Precision (mAP) for a set of queries.
#     :param retrieved_lists: List of lists, each containing retrieved item indices for a query.
#     :param relevant_lists: List of lists, each containing relevant item indices for a query.
#     :return: Mean Average Precision score.
#     """
#     ap_scores = [average_precision(retrieved, relevant)
#                  for retrieved, relevant in zip(retrieved_lists, relevant_lists)]
    
#     return np.mean(ap_scores)

def mean_average_precision(predictions, retrieval_solution, max_predictions=100):
    """Computes mean average precision for retrieval prediction.
  Args:
    predictions: Dict mapping test image ID to a list of strings corresponding
      to index image IDs.
    retrieval_solution: Dict mapping test image ID to list of ground-truth image
      IDs.
    max_predictions: Maximum number of predictions per query to take into
      account. For the Google Landmark Retrieval challenge, this should be set
      to 100.
  Returns:
    mean_ap: Mean average precision score (float).
  Raises:
    ValueError: If a test image in `predictions` is not included in
      `retrieval_solutions`.
  """
    # Compute number of test images.
    num_test_images = len(retrieval_solution.keys())

    # Loop over predictions for each query and compute mAP.
    mean_ap = 0.0
    for key, prediction in predictions.items():
        if key not in retrieval_solution:
            raise ValueError(
                'Test image %s is not part of retrieval_solution' % key)

        # Loop over predicted images, keeping track of those which were already
        # used (duplicates are skipped).
        ap = 0.0
        already_predicted = set()
        num_expected_retrieved = min(
            len(retrieval_solution[key]), max_predictions)
        num_correct = 0
        for i in range(min(len(prediction), max_predictions)):
            if prediction[i] not in already_predicted:
                if prediction[i] in retrieval_solution[key]:
                    num_correct += 1
                    ap += num_correct / (i + 1)
                already_predicted.add(prediction[i])

        ap /= num_expected_retrieved
        mean_ap += ap

    mean_ap /= num_test_images

    return mean_ap

def mean_recall_at_k(predictions, retrieval_solution, k):
    """Computes mean recall at K for retrieval prediction.
    Args:
        predictions: Dict mapping test image ID to a list of strings corresponding
            to index image IDs.
        retrieval_solution: Dict mapping test image ID to list of ground-truth image
            IDs.
        k: The number of top predictions to consider for calculating recall.
    Returns:
        mean_recall: Mean recall at K score (float).
    Raises:
        ValueError: If a test image in `predictions` is not included in
            `retrieval_solution`.
    """
    num_test_images = len(retrieval_solution.keys())
    total_recall = 0.0

    for key, prediction in predictions.items():
        if key not in retrieval_solution:
            raise ValueError('Test image %s is not part of retrieval_solution' % key)

        relevant_items = set(retrieval_solution[key])
        top_k_predictions = prediction[:k]
        num_relevant_at_k = len([pred for pred in top_k_predictions if pred in relevant_items])
        recall_at_k = num_relevant_at_k / len(relevant_items) if relevant_items else 0
        total_recall += recall_at_k

    mean_recall = total_recall / num_test_images
    return mean_recall

# def recall_at_k(retrieved, relevant, k):
#     """
#     Calculate Recall at K for a single query.
#     :param retrieved: List of retrieved item indices.
#     :param relevant: List of relevant item indices.
#     :param k: The number of top results to consider.
#     :return: Recall at K score.
#     """
#     retrieved_top_k = set(retrieved[:k])
#     relevant_set = set(relevant)
#     recall = len(retrieved_top_k.intersection(relevant_set)) / len(relevant_set)
#     return recall

# def mean_recall_at_k(retrieved_lists, relevant_lists, k):
#     """
#     Calculate Mean Recall at K for a set of queries.
#     :param retrieved_lists: List of lists, each containing retrieved item indices for a query.
#     :param relevant_lists: List of lists, each containing relevant item indices for a query.
#     :param k: The number of top results to consider for each query.
#     :return: Mean Recall at K score.
#     """
#     recall_scores = [recall_at_k(retrieved, relevant, k)
#                      for retrieved, relevant in zip(retrieved_lists, relevant_lists)]
    
#     return np.mean(recall_scores)





# from github: https://github.com/filipradenovic/revisitop/blob/master/python/evaluate.py


def compute_ap(ranks, nres):
    """
    Computes average precision for given ranked indexes.
    
    Arguments
    ---------
    ranks : zero-based ranks of positive images
    nres  : number of positive images
    
    Returns
    -------
    ap    : average precision
    """

    # number of images ranked by the system
    nimgranks = len(ranks)

    # accumulate trapezoids in PR-plot
    ap = 0

    recall_step = 1. / nres

    for j in np.arange(nimgranks):
        rank = ranks[j]

        if rank == 0:
            precision_0 = 1.
        else:
            precision_0 = float(j) / rank

        precision_1 = float(j + 1) / (rank + 1)

        ap += (precision_0 + precision_1) * recall_step / 2.

    return ap

def compute_map(ranks, gnd, kappas=[]):
    """
    Computes the mAP for a given set of returned results.

         Usage: 
           map = compute_map (ranks, gnd) 
                 computes mean average precsion (map) only
        
           map, aps, pr, prs = compute_map (ranks, gnd, kappas) 
                 computes mean average precision (map), average precision (aps) for each query
                 computes mean precision at kappas (pr), precision at kappas (prs) for each query
        
         Notes:
         1) ranks starts from 0, ranks.shape = db_size X #queries
         2) The junk results (e.g., the query itself) should be declared in the gnd stuct array
         3) If there are no positive images for some query, that query is excluded from the evaluation
    """

    map = 0.
    nq = len(gnd) # number of queries
    aps = np.zeros(nq)
    pr = np.zeros(len(kappas))
    prs = np.zeros((nq, len(kappas)))
    nempty = 0

    for i in np.arange(nq):
        qgnd = np.array(gnd[i]['ok'])

        # no positive images, skip from the average
        if qgnd.shape[0] == 0:
            aps[i] = float('nan')
            prs[i, :] = float('nan')
            nempty += 1
            continue

        try:
            qgndj = np.array(gnd[i]['junk'])
        except:
            qgndj = np.empty(0)

        # sorted positions of positive and junk images (0 based)
        pos  = np.arange(ranks.shape[0])[np.in1d(ranks[:,i], qgnd)]
        junk = np.arange(ranks.shape[0])[np.in1d(ranks[:,i], qgndj)]

        k = 0;
        ij = 0;
        if len(junk):
            # decrease positions of positives based on the number of
            # junk images appearing before them
            ip = 0
            while (ip < len(pos)):
                while (ij < len(junk) and pos[ip] > junk[ij]):
                    k += 1
                    ij += 1
                pos[ip] = pos[ip] - k
                ip += 1

        # compute ap
        ap = compute_ap(pos, len(qgnd))
        map = map + ap
        aps[i] = ap

        # compute precision @ k
        pos += 1 # get it to 1-based
        for j in np.arange(len(kappas)):
            kq = min(max(pos), kappas[j]); 
            prs[i, j] = (pos <= kq).sum() / kq
        pr = pr + prs[i, :]

    map = map / (nq - nempty)
    pr = pr / (nq - nempty)

    return map, aps, pr, prs