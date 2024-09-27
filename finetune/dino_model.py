import torch.nn as nn
import torch


class DinoVisionTransformerClassifier(nn.Module):
    def __init__(self, hub_path=None, ssl_dino_path=None, num_classes=1):
        super(DinoVisionTransformerClassifier, self).__init__()
        if not ssl_dino_path:
            self.model = torch.hub.load('facebookresearch/dinov2',
                                        hub_path)
        else:
            self.model = get_dino_finetuned_downloaded(ssl_dino_path)

        embeddings_size = self.model.norm.weight.shape[0]

        self.classifier = nn.Sequential(
            nn.Linear(embeddings_size, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.model(x)
        x = self.model.norm(x)
        x = self.classifier(x)
        return x


def get_dino_finetuned_downloaded(dino_path_finetuned):
    model=torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')

    pretrained = torch.load(dino_path_finetuned, map_location=torch.device('cpu'))
    # make correct state dict for loading
    new_state_dict = {}
    for key, value in pretrained['teacher'].items():
        if 'dino_head' in key:
            print('not used')
        else:
            new_key = key.replace('backbone.', '')
            new_state_dict[new_key] = value
    #change shape of pos_embed, shape depending on vits or vitg
    pos_embed = nn.Parameter(torch.zeros(1, 257, 384))
    model.pos_embed = pos_embed
    # load state dict
    model.load_state_dict(new_state_dict, strict=True)
    return model





