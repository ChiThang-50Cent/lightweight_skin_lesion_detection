import torch
import torchvision


class Models:
    def __init__(
        self, model_name, num_classes, feature_extract, pre_trained=True
    ) -> None:
        self.model_name = model_name
        self.pre_trained = pre_trained
        self.feature_extract = feature_extract
        self.num_classes = num_classes

    def set_parameter_requires_grad(self, model: torch.nn.Module):
        if self.feature_extract:
            for param in model.parameters():
                param.requires_grad = False

    def get_model(self) -> torch.nn.Module:
        model = None

        if self.model_name == "efficientnet":
            weight = (
                None
                if not self.pre_trained
                else torchvision.models.EfficientNet_B0_Weights
            )

            model = torchvision.models.efficientnet_b0(weights=weight)
            self.set_parameter_requires_grad(model)
            num_ftrs = model.classifier[-1].in_features
            model.classifier[-1] = torch.nn.Linear(num_ftrs, self.num_classes)

        elif self.model_name == "mobilenetv3":
            weight = (
                None
                if not self.pre_trained
                else torchvision.models.MobileNet_V3_Large_Weights
            )

            model = torchvision.models.mobilenet_v3_large(weights=weight)
            self.set_parameter_requires_grad(model)
            num_ftrs = model.classifier[-1].in_features
            model.classifier[-1] = torch.nn.Linear(num_ftrs, self.num_classes)

        elif self.model_name == "shufflenetv2":
            weight = (
                None
                if not self.pre_trained
                else torchvision.models.ShuffleNet_V2_X2_0_Weights
            )
            model = torchvision.models.shufflenet_v2_x2_0(weights=weight)
            self.set_parameter_requires_grad(model)
            num_ftrs = model.fc.in_features
            model.fc = torch.nn.Linear(num_ftrs, self.num_classes)
        
        elif self.model_name == "resnet":
            weight = (
                None
                if not self.pre_trained
                else torchvision.models.ResNet18_Weights.DEFAULT
            )
            model = torchvision.models.resnet18(weights=weight)
            self.set_parameter_requires_grad(model)
            num_ftrs = model.fc.in_features
            model.fc = torch.nn.Linear(num_ftrs, self.num_classes)
        
        elif self.model_name == "mnasnet":
            weight = (
                None
                if not self.pre_trained
                else torchvision.models.MNASNet1_3_Weights.DEFAULT
            )
            model = torchvision.models.mnasnet1_3(weights=weight)
            self.set_parameter_requires_grad(model)
            num_ftrs = model.classifier[-1].in_features
            model.classifier[-1] = torch.nn.Linear(num_ftrs, self.num_classes)

        else:
            print("Invalid model name, exiting...")
            exit()
        return model
