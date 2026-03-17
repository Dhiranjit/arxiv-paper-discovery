import torch
import torch.nn as nn


class SciBERTClassifier(nn.Module):
    def __init__(self, llm, dropout_p, num_classes):
        super().__init__()
        self.llm = llm
        self.dropout = nn.Dropout(dropout_p)
        self.classifier = nn.Linear(llm.config.hidden_size, num_classes)

    def forward(self, x):
        input_ids = x["input_ids"]
        attention_mask = x["attention_mask"]

        token_type_ids = x.get("token_type_ids", None)

        output = self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        z = self.dropout(output.pooler_output)
        z = self.classifier(z)

        return z