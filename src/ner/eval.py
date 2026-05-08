# 定义一个训练器
import seqeval
from datasets import load_dataset,load_from_disk#加载数据
from transformers import AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification, \
    EvalPrediction, EarlyStoppingCallback
from configuration.config import *
from transformers import Trainer, TrainingArguments
import time
import evaluate
from transformers import Trainer,AutoTokenizer
from configuration.config import *



tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

#模型
model = AutoModelForTokenClassification.from_pretrained(CHECKPOINT_DIR/NER_DIR/'best_model')

#3.加载数据集
test_dataset = load_from_disk(PROCESSED_DATA_DIR/'test')
#4.数据整理器
data_collator = DataCollatorForTokenClassification(

    tokenizer=tokenizer,
    padding=True,
    return_tensors="pt",
    #label_pad_token_id=
)

seqeval = evaluate.load('seqeval')

def compute_metrics(prediction:EvalPrediction):
    #提取模型的预测输出和真实标签
    logits = prediction.predictions
    preds = logits.argmax(axis=-1)#预测分类标签
    labels = prediction.label_ids
    # 将标签id转换成真正的标注标签BIO
    unpad_labels=[]
    unpad_preds=[]
    for pred,label in zip(preds,labels):
        #去掉填充对应的id
        unpad_label= label[label!=-100]
        unpad_pred = pred[label!=-100]
        #转BIO标签
        unpad_pred = [model.config.id2label[id] for id in unpad_pred]
        unpad_label =[model.config.id2label[id] for id in unpad_label]
        #添加到列表
        unpad_labels.append(unpad_label)
        unpad_preds.append(unpad_pred)
    result =  seqeval.compute(predictions=unpad_preds,references=unpad_labels)
    return result
early_stopping_callback =EarlyStoppingCallback(early_stopping_patience=2)
trainer=Trainer(
    model=model,
    eval_dataset=test_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)
#创建训练器

if __name__ == "__main__":
    result = trainer.evaluate()
    print(result)