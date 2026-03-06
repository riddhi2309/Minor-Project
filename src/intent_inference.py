import torch 
from transformers import RobertaTokenizer, AutoModelForSequenceClassification 
MODEL_PATH = "models/intent_classifier" 
class IntentClassifier: 
    def __init__(self): 
        self.tokenizer = RobertaTokenizer.from_pretrained( MODEL_PATH, use_fast=False ) 
        self.model = AutoModelForSequenceClassification.from_pretrained( MODEL_PATH ) 
        self.model.eval() 
    def predict(self, text): 
        inputs = self.tokenizer( text, return_tensors="pt", truncation=True, padding=True ) 
        with torch.no_grad(): outputs = self.model(**inputs) 
        probs = torch.softmax(outputs.logits, dim=1) 
        confidence, label_id = torch.max(probs, dim=1) 
        return { "intent": self.model.config.id2label[label_id.item()], "confidence": float(confidence) }