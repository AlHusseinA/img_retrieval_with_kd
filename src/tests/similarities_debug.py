import sys
ROOT="/home/alabutaleb/Desktop/myprojects/compression_retrieval_proj/mvp"
sys.path.append(f"{ROOT}/src")
print(sys.path)

import unittest
import torch
from torchmetrics.retrieval import RetrievalMAP, RetrievalRecall
from torchvision.models import ResNet50_Weights
from models.resnet50_conv_compressionV2 import ResNet50_convV2
from dataloaders.cub200loader import DataLoaderCUB200
from utils.similarities import image_retrieval, create_gallery_features

class TestImageRetrievalSystem(unittest.TestCase):

    def test_cosine_similarity(self):
        query_feature = torch.randn(1, 2048)
        gallery_features = torch.randn(10, 2048)
        expected_shape = (10,)

        similarity_scores, _, _ = image_retrieval(query_feature, gallery_features)
        self.assertEqual(similarity_scores.shape, expected_shape)

    def test_feature_dimension_validation(self):
        query_feature = torch.randn(1, 2048)
        gallery_features = torch.randn(10, 1024)

        with self.assertRaises(ValueError):
            image_retrieval(query_feature, gallery_features)

    def test_gallery_feature_creation(self):
        load_dir = f"/home/alabutaleb/Desktop/confirmation/baselines_allsizes/weights"
        data_root ="/media/alabutaleb/data/cub200/"

        dataloadercub200 = DataLoaderCUB200(data_root, batch_size=256, num_workers=10)
        trainloader_cub200, testloader_cub200 = dataloadercub200.get_dataloaders()

        model_cub200 = ResNet50_convV2(2048, 200, weights=ResNet50_Weights.DEFAULT, pretrained_weights=None)

        fine_tuned_weights = torch.load(f'{load_dir}/resnet50_feature_size_2048_cub200_batchsize_256_lr_7e-05.pth')
        # save_dir = "/media/alabutaleb/09d46f11-3ed1-40ce-9868-932a0133f8bb/data/resnet50_finetuned/experiment_gpu_1/weights"
        model_cub200.load_state_dict(fine_tuned_weights)
        model_cub200.feature_extractor_mode()

        # You need to provide mock_model and mock_trainloader
        gallery_features, gallery_labels = create_gallery_features(model_cub200, trainloader_cub200)
        self.assertEqual(len(gallery_features), len(gallery_labels), "Mismatch in number of gallery features and labels.")
        pass  # Replace with actual test when mock objects are available

    def test_retrieval_metrics_initialization(self):
        rmap = RetrievalMAP()
        r1 = RetrievalRecall(top_k=1)
        self.assertIsNotNone(rmap, "RetrievalMAP was not initialized correctly.")
        self.assertIsNotNone(r1, "RetrievalRecall was not initialized correctly.")

    def test_ground_truth_calculation(self):
        query_label = torch.tensor([3])  # Assuming label '3' for the query
        gallery_labels = torch.tensor([1, 2, 3, 4, 5])  # Simulated gallery labels
        expected_ground_truths = torch.tensor([0, 0, 1, 0, 0])  # Expected ground truths

        ground_truths = (gallery_labels == query_label).int()
        self.assertTrue(torch.equal(ground_truths, expected_ground_truths), "Ground truth calculation is incorrect.")

if __name__ == '__main__':
    unittest.main()




