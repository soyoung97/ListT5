import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import jsonlines

model_path = 'castorini/duot5-base-msmarco-10k'
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path).to('cuda')

data = []
with jsonlines.open('./fiqa_input.jsonl', 'r') as line:
    for x in line:
        data.append(x)

pred_changed_cnt = 0
total_cnt = 0
accuracy = [0, 0]
for data_source in data:
  for i, negative_text in enumerate(data_source['negs']):
    posfirst = f"Query: {data_source['question']} Document0: {data_source['pos']} Document1: {negative_text} Relevant:"
    posfirst_tensors = tokenizer(posfirst, return_tensors='pt', padding='max_length', max_length=512, truncation=True).to('cuda')
    posfirst_output = model.generate(**posfirst_tensors, max_length=2, return_dict_in_generate=True, output_scores=True)
    posfirst_output_scores = torch.nn.functional.log_softmax(posfirst_output.scores[0][:, [1176, 6136]][0], dim=0)
    posfirst_pos, posfirst_neg = posfirst_output_scores
    if posfirst_pos > posfirst_neg:
        accuracy[0] += 1

    negfirst = f"Query: {data_source['question']} Document0: {negative_text} Document1: {data_source['pos']} Relevant:"
    negfirst_tensors = tokenizer(negfirst, return_tensors='pt', padding='max_length', max_length=512, truncation=True).to('cuda')
    negfirst_output = model.generate(**negfirst_tensors, max_length=2, return_dict_in_generate=True, output_scores=True)
    negfirst_output_scores = torch.nn.functional.log_softmax(negfirst_output.scores[0][:, [1176, 6136]][0], dim=0)
    negfirst_neg, negfirst_pos = negfirst_output_scores
    if negfirst_neg < negfirst_pos:
        accuracy[1] += 1

    if (posfirst_pos > posfirst_neg and negfirst_pos < negfirst_neg) or (posfirst_pos < posfirst_neg and negfirst_pos > negfirst_neg):
      pred_changed = '%%%%%%%%%%%%%%Prediction changed by order!!%%%%%%%%%%%%%%%%'
      pred_changed_cnt += 1
    else:
      pred_changed = ''

    print(f"idx: {i} {pred_changed}")
    print(f"Pos in posfirst: {posfirst_pos:2f}, in negfirst: {negfirst_pos:2f}, diff {abs(posfirst_pos - negfirst_pos):2f}")
    print(f"Neg in posfirst: {posfirst_neg:2f}, in negfirst: {negfirst_neg:2f}, diff {abs(posfirst_neg - negfirst_neg):2f}")
    print()
    total_cnt += 1

print(f'-------------- prediction change ratio: {100 * (pred_changed_cnt / total_cnt)}% -----------------')
print(f"accuracy: 1st {accuracy[0]/total_cnt}, 2nd {accuracy[1]/total_cnt}, total {total_cnt}")
import pdb; pdb.set_trace()
