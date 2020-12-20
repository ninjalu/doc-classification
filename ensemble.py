import torch
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader
from typing import Dict, List, Tuple, Optional, Union
from transformers import PreTrainedTokenizerBase
from sklearn.model_selection import train_test_split

class NLPEnsemble():
    """
    This class takes in traditional and neural NLP models, make predictions given the model power ratio.
    """
    def __init__(self, traditional_nlp, nn_nlp, ratio=0.5):
        self.traditional_nlp = traditional_nlp
        self.nn_nlp = nn_nlp
        self.ratio = ratio
        self.t_nlp_pred = None
        self.nn_nlp_pred = None
    
    def fit(self, X_train, y_train):
        self.traditional_nlp.fit(X_train, y_train)
    
    def predict(self, X_test, test_dataloader):
        self.t_nlp_pred = self.traditional_nlp.predict(X_test, test_dataloader)
        self.nn_nlp_pred = _nn_predict(self.nn_nlp, test_dataloader)
        assert len(self.t_nlp_pred) == len(self.nn_nlp_pred)
        self.pred = self.ratio*self.t_nlp_pred + (1-self.ratio)*self.nn_nlp_pred
        return self.pred

def _nn_predict(model, test_dataloader):
    pred = []
    model.eval()
    for batch in test_dataloader:
        input_ids, attention_masks, labels = batch
        with torch.no_grad():
            output = model(
                input_ids=input_ids,
                token_type_ids=None,
                attention_masks=attention_masks,
                labels=labels
            )
            logits = output[1]
        pred.append(torch.argmax(logits, dim=1).numpy())
    return pred
