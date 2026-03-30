import torch
from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0 ,EfficientNet_B3_Weights, efficientnet_b3 ,EfficientNet_B7_Weights, efficientnet_b7, EfficientNet_V2_L_Weights, efficientnet_v2_l

# Create modelbuilder function for effnet_b0 model
def create_effnetb0_model(out_features:int, dropout:float, device:str) -> nn.Module:
    print("----------- EffnetB0 model creation -----------\n")
    print(f"Creating effnet_b0 model with {out_features} out features and a dropout of {dropout}...")
    # Get the default weights
    effnet_b0_weights = EfficientNet_B0_Weights.DEFAULT

    # Create the model
    effnet_b0_model = efficientnet_b0(weights=effnet_b0_weights)

    # Deactivate the grad on each parameter
    for parameter in effnet_b0_model.features.parameters():
        parameter.requires_grad = False
    
    # Create own classifier
    effnet_b0_model.classifier = nn.Sequential(
        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(in_features=1280,
              out_features=out_features,
              bias=True).to(device)
    )
    # Send model to target device
    effnet_b0_model.to(device)

    # Return model and weights
    print("Effnet_B0 model created!\n")
    return effnet_b0_model, effnet_b0_weights

# Create modelbuilder function for effnet_b3 model
def create_effnetb3_model(out_features:int, dropout:float, device:str) -> nn.Module:
    print("----------- EffnetB3 model creation -----------\n")
    print(f"Creating effnet_b3 model with {out_features} out features and a dropout of {dropout}...")
    # Get the default weights
    effnet_b3_weights = EfficientNet_B3_Weights.DEFAULT

    # Create the model
    effnet_b3_model = efficientnet_b3(weights=effnet_b3_weights)

    # Deactivate the grad on each parameter
    for parameter in effnet_b3_model.features.parameters():
        parameter.requires_grad = False
    
    # Create own classifier
    effnet_b3_model.classifier = nn.Sequential(
        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(in_features=1536,
              out_features=out_features,
              bias=True).to(device)
    )
    # Send model to target device
    effnet_b3_model.to(device)

    # Return model and weights
    print("Effnet_3b model created!\n")
    return effnet_b3_model, effnet_b3_weights

# Create modelbuilder function for effnet_b7 model
def create_effnetb7_model(out_features:int, dropout:float, device:str) -> nn.Module:
    print("----------- EffnetB7 model creation -----------\n")
    print(f"Creating effnet_b7 model with {out_features} out features and a dropout of {dropout}...")
    # Get the default weights
    effnet_b7_weights = EfficientNet_B7_Weights.DEFAULT

    # Create the model
    effnet_b7_model = efficientnet_b7(weights=effnet_b7_weights)

    # Deactivate the grad on each parameter
    for parameter in effnet_b7_model.features.parameters():
        parameter.requires_grad = False
    
    # Create own classifier
    effnet_b7_model.classifier = nn.Sequential(
        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(in_features=2560,
              out_features=out_features,
              bias=True).to(device)
    )
    # Send model to target device
    effnet_b7_model.to(device)

    # Return model and weights
    print("Effnet_7b model created!\n")
    return effnet_b7_model, effnet_b7_weights

# Create modelbuilder function for effnet_v2_l model
def create_effnet_v2_l_model(out_features:int, dropout:float, device:str) -> nn.Module:
    print("----------- EffnetV2L model creation -----------\n")
    print(f"Creating effnet_v2_l model with {out_features} out features and a dropout of {dropout}...")
    # Get the default weights
    effnet_v2_l_weights = EfficientNet_V2_L_Weights.DEFAULT

    # Create the model
    effnet_v2_l_model = efficientnet_v2_l(weights=effnet_v2_l_weights)

    # Deactivate the grad on each parameter
    for parameter in effnet_v2_l_model.features.parameters():
        parameter.requires_grad = False
    
    # Create own classifier
    effnet_v2_l_model.classifier = nn.Sequential(
        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(in_features=1280,
              out_features=out_features,
              bias=True).to(device)
    )
    # Send model to target device
    effnet_v2_l_model.to(device)

    # Return model and weights
    print("Effnet_v2_l model created!\n")
    return effnet_v2_l_model, effnet_v2_l_weights
    


