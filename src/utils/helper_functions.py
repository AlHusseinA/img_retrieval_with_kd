import torch
from copy import deepcopy

def check_size(model, input_size=(3, 224, 224)):
    """
    Check the size of the model output
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Check if the model is wrapped in DataParallel and unwrap if necessary
    if isinstance(model, torch.nn.DataParallel):
        unwrapped_model = model.module
    else:
        unwrapped_model = model

    model_copy = deepcopy(unwrapped_model).to(device)
    model_copy.feature_extractor_mode()

    # Generate a random input tensor of the specified size and move it to the same device as the model
    input_tensor = torch.randn(1, *input_size).to(device)
    output_size = model_copy(input_tensor).size()[1]

    # print(f"Input size: {input_size}")
    # print(f"Output size: {output_size}")

    return output_size



# import torch
# from copy import deepcopy


# def check_size(model, input_size=( 3, 224, 224)):
#     """
#     Check the size of the model output
#     """
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model_copy = deepcopy(model).to(device)
#     model_copy.feature_extractor_mode()
#     input = torch.randn(1, *input_size).to(device)
#     output_size = model_copy(input).size()[1]
#     # output_size = output_size.to(device)
#     print(f"Input size: {input_size}")
#     print(f"Output size: {output_size}")    # Move input to the same device as the model

#     return output_size
    
   

#    def check_size(model, input_size=(3, 224, 224)):
#     model_copy = copy.deepcopy(model).to(device)
#     random_input = torch.randn(1, *input_size).to(device)  # Move input to the same device as the model
#     output_size = model_copy(random_input).size()
#     print(f"Output size: {output_size}")
#     return output_size
