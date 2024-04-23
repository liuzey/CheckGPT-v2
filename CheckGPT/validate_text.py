import torch
import time
import json
import argparse
from transformers import RobertaTokenizer, RobertaModel, RobertaConfig, RobertaPreTrainedModel
import torch.nn.functional as F
from model import CheckGPT


parser = argparse.ArgumentParser(description='Demo of argparse')
parser.add_argument('data', type=str)
parser.add_argument('model', type=str)
parser.add_argument('gpt', type=int, default=1)
parser.add_argument('--number', type=int, default=0)
parser.add_argument('--interval', type=int, default=25)
parser.add_argument('--v1', type=int, default=0)
args = parser.parse_args()


class CustomRobertaForPipeline(RobertaPreTrainedModel):
    def __init__(self, config, device="cuda"):
        super().__init__(config)
        self.roberta = RobertaModel(config)
        self.classifier = CheckGPT(input_size=1024, hidden_size=256, batch_first=True, dropout=0.5, bidirectional=True,
                              num_layers=2, device=device, v1=args.v1)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None):
        outputs = self.roberta(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                                   position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds)
        features = F.pad(outputs.last_hidden_state, (0, 0, 0, 512 - outputs.last_hidden_state.size(1)))
        logits = self.classifier(features, lengths=torch.tensor([outputs.last_hidden_state.size(1)]))
        return logits

device_name = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_name)
tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
config = RobertaConfig.from_pretrained("roberta-large", num_labels=2)
model = CustomRobertaForPipeline.from_pretrained("roberta-large", config=config, device=device_name)
try:
    model.classifier.load_state_dict(torch.load(args.model), strict=False)
except RuntimeError:
    model.classifier = CheckGPT(input_size=1024, hidden_size=128, batch_first=True, dropout=0.5, bidirectional=True,
                                num_layers=2, device=device_name, v1=args.v1)
    model.classifier.load_state_dict(torch.load(args.model), strict=False)
model = model.to(device)
model.eval()
print("Test model: {}, data: {}, is gpt: {}".format(args.model, args.data, args.gpt))


def eval_one(model, input):
    item = input.replace("\n", " ").replace("  ", " ").strip()
    tokens = tokenizer.encode(item, truncation=True, max_length=512)
    outputs = model(torch.tensor(tokens).unsqueeze(0).to(device))
    pred = torch.max(outputs.data, 1)[1]
    (gpt_prob, hum_prob) = F.softmax(outputs.data, dim=1)[0]
    return pred[0].data, 100 * gpt_prob, 100 * hum_prob

with open(args.data, "r") as f:
    data = json.load(f)
f.close()

correct, total, length = 0, 0, args.number if args.number else len(data)
start = time.time()
for i in range(length):
    try:
        input_text = list(data.values())[i]["abstract"]
        pred, gpt_prob, hum_prob = eval_one(model, input_text)
        if pred == 1 - args.gpt:
            correct += 1
        total += 1
        if (i > 0 and i % args.interval == 0) or i == length - 1:
            print("{}/{}: Accuracy: {:.2f}%".format(i, length, correct/(total + 1e-6) * 100))

    except KeyboardInterrupt:
        exit()
