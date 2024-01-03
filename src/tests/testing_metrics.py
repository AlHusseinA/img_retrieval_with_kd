# from ..utils import metrics
import sys
sys.path.append('/home/alabutaleb/Desktop/myprojects/compression_retrieval_proj/mvp/src')
print(sys.path)


from utils.metrics import calculate_map, compute_metrics
import numpy as np

def test_calculate_map():
    # Case 1: General case with a variety of prediction values
    predictions1 = [
        [0.9, 0.8, 0.7, 0.6, 0.5],
        [0.5, 0.6, 0.7, 0.8, 0.9],
        [0.1, 0.2, 0.3, 0.4, 0.5],
    ]
    ground_truths1 = [
        [1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0],
    ]
    map_score1 = calculate_map(predictions1, ground_truths1)
    assert 0 <= map_score1 <= 1, "mAP score must be between 0 and 1"
    # Use a calculated expected mAP value based on the inputs
    print(f"mAP score is {map_score1}")
    assert map_score1 == 0.6666666666666666, "mAP score should be 0.8167 for the given inputs"
    
    # Case 2: All predictions are correct
    predictions2 = [
        [0.9, 0.8],
        [0.7, 0.6],
    ]
    ground_truths2 = [
        [1, 1],
        [1, 1],
    ]
    map_score2 = calculate_map(predictions2, ground_truths2)
    assert map_score2 == 1.0, "mAP score should be 1.0 when all predictions are correct"
    
    # Case 3: All predictions are incorrect
    predictions3 = [
        [0.1, 0.2],
        [0.3, 0.4],
    ]
    ground_truths3 = [
        [0, 0],
        [0, 0],
    ]
    map_score3 = calculate_map(predictions3, ground_truths3)
    assert map_score3 == 1.0, "mAP score should be 1.0 when there are no relevant items in the ground truths"
    
    # Case 4: Different sized inputs
    predictions4 = [
        [0.9],
        [0.7, 0.6],
    ]
    ground_truths4 = [
        [1],
        [1, 0],
    ]
    map_score4 = calculate_map(predictions4, ground_truths4)
    assert map_score4 == 1.0, "mAP score should be 1.0 for the given inputs with different sizes"

    print("calculate_map passed")




def test_compute_metrics():

    sorted_scores = np.array([0.9, 0.8, 0.7, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0, 0.0])
    sorted_indices = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    query_label = 1
    gallery_labels = np.array([1, 1, 0, 0, 1, 0, 0, 1, 0, 0])

    ap, precision_at_k, recall_at_k = compute_metrics(sorted_scores, sorted_indices, query_label, gallery_labels)


    # Define the expected values based on the input data
    expected_precision_at_k = {1: 1.0, 5: 0.6, 10: 0.4}
    expected_recall_at_k = {1: 0.25, 5: 0.75, 10: 1.0}
    expected_ap = 0.4375  # Manually computed expected AP value
    assert 0 <= ap <= 1, "AP must be between 0 and 1"
    assert ap == expected_ap, f"AP should be {expected_ap}"


    for k in [1, 5, 10]:
        assert 0 <= precision_at_k[k] <= 1, f"P@{k} must be between 0 and 1"
        assert precision_at_k[k] == expected_precision_at_k[k], f"P@{k} should be {expected_precision_at_k[k]}"
        
        assert 0 <= recall_at_k[k] <= 1, f"R@{k} must be between 0 and 1"
        assert recall_at_k[k] == expected_recall_at_k[k], f"R@{k} should be {expected_recall_at_k[k]}"

    print("compute_metrics passed")





if __name__ == "__main__":
    test_calculate_map()
    test_compute_metrics()
 