from torch import nn
import torch

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
    def forward(self, x):
        return self.net(x)


def preprocess(document_content, vectorizer, lsa):
    if isinstance(document_content, str):
        inputs = vectorizer.transform([document_content])
    else:
        inputs = vectorizer.transform(document_content)
    inputs = lsa.transform(inputs)
    return inputs
    
def predict_mlp(document_content, vectorizer, lsa, model):

    inputs = preprocess(document_content, vectorizer, lsa)
    
    with torch.no_grad():    
        logits = model(torch.from_numpy(inputs).float())
    
    return logits