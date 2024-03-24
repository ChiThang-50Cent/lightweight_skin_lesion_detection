import torch
import torchvision


class Models:
    def __init__(
        self, model_name, num_classes, 
        feature_extract, pre_trained=True
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
            model = torchvision.models.efficientnet_v2_s(pretrained=self.pre_trained)
            self.set_parameter_requires_grad(model)
            num_ftrs = model.classifier[-1].in_features
            model.classifier[-1] = torch.nn.Linear(num_ftrs, self.num_classes)

        elif self.model_name == "mobilenetv3":
            model = torchvision.models.mobilenet_v3_large(pretrained=self.pre_trained)
            self.set_parameter_requires_grad(model)
            num_ftrs = model.classifier[-1].in_features
            model.classifier[-1] = torch.nn.Linear(num_ftrs, self.num_classes)

        elif self.model_name == "shuffernetv2":
            model = torchvision.models.shufflenet_v2_x2_0(pretrained=self.pre_trained)
            self.set_parameter_requires_grad(model)
            num_ftrs = model.fc.in_features
            model.fc = torch.nn.Linear(num_ftrs, self.num_classes)
        else:
            print("Invalid model name, exiting...")
            exit()
        return model
