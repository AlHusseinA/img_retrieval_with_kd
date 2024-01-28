
import sys
import os

ROOT="/home/alabutaleb/Desktop/myprojects/compression_retrieval_proj/mvp"
sys.path.append(f"{ROOT}/src")
print(f"{sys.path=}")
print(f"{os.getcwd()=}")
from dataloaders.cub200loader import DataLoaderCUB200
from models.resnet50_vanilla import ResNet50_vanilla
from torchvision.models import resnet50, ResNet50_Weights
from copy import deepcopy
import torch
from tqdm import tqdm
from utils.features_unittest import TestFeatureSize

def extract_features(model, dataloader, device=None):
    """
    Passes data through the feature extractor and accumulates the features in a tensor.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    features_list = []

    with torch.inference_mode():
        for images, _ in tqdm(dataloader, desc="Extracting features"):
            images = images.to(device)
            features = model(images)
            features_list.append(features.cpu())

    features = torch.vstack(features_list)
    return features

def save_features(model, dataloader, filename, save_dir, device=None):
    """
    Saves the features extracted from the dataloader to a file in the specified directory.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    features = extract_features(model, dataloader, device)
    file_path = os.path.join(save_dir, filename)
    torch.save(features, file_path)



def load_resnet50_unmodifiedVanilla(num_classes_cub200,feature_size,load_dir):

    model = ResNet50_vanilla(num_classes_cub200, weights=ResNet50_Weights.DEFAULT, pretrained_weights=None)        
    fine_tuned_weights = torch.load(f'{load_dir}/resnet50_feature_size_vanilla_cub200_batchsize_256_lr_7e.pth')
    # fine_tuned_weights = torch.load(f'{load_dir}/resnet50_feature_size_vanilla_cub200_batchsize_256_lr_7e.pth')

    model.load_state_dict(fine_tuned_weights)
    # model.feature_extractor_mode()
    #unit test for feature size
    test = deepcopy(model)
    testing_size = TestFeatureSize(test, feature_size) # this will confirm that the feature size is correct

    assert model is not None, "Failed to load the model"
    try:
        testing_size.test_feature_size()
        print(f"The loaded model under evaluation is in indeed with {feature_size} feature size!")

    except AssertionError as e:
        # add an error message to the assertion error
        e.args += (f"Expected feature size {feature_size}, got {test.features_out.in_features}")   
        raise e # if the feature size is not correct, raise an error
    

    return model

def main():
    """
    Main function to process and save features of train and test splits.
    """
    print(f"####"*20)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"####"*20)
    print(f"Generating features for train and test splits using an unmodified vanilla Resnet50 Feature Extractor")
    print(f"####"*20)
    # load data
    #### get data #####root, batch_size=32,num_workers=10   
    data_root ="/media/alabutaleb/data/cub200/"
    batch_size = 256
    dataloadercub200 = DataLoaderCUB200(data_root, batch_size=batch_size, num_workers=10)
    num_classes_cub200 = dataloadercub200.get_number_of_classes()
    trainloader_cub200, testloader_cub200 = dataloadercub200.get_dataloaders()

    # load model
    load_dir_vanilla = f"/home/alabutaleb/Desktop/confirmation"    # "resnet50_feature_size_vanilla_cub200_batchsize_256_lr_7e.pth"
    feature_size_unmodifed = 2048
    model = load_resnet50_unmodifiedVanilla(num_classes_cub200, feature_size_unmodifed, load_dir_vanilla)
    model.to(device)
    model.feature_extractor_mode()
    print(f"Model loaded from {load_dir_vanilla}")



    # save dir
    save_dir = "/home/alabutaleb/Desktop/confirmation/features_vanilla_only"
    save_features(model, trainloader_cub200, "train_features.pth", save_dir)
    save_features(model, testloader_cub200, "test_features.pth", save_dir)





if __name__ == "__main__":
    # Assuming model, trainloader, and testloader are defined elsewhere
    # main(model, trainloader, testloader

    main()
