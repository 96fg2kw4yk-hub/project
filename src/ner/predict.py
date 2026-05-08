#自定义预测期类
#from accelerate.test_utils.scripts.external_deps.test_ds_alst_ulysses_sp import outputs
from transformers  import AutoTokenizer,AutoModelForTokenClassification
from torch._C import device
import torch
from configuration.config import *
class Predictor:
    def __init__(self,model,tokenizer,device):
         self.model = model.to(device)
         self.model.eval()
         self.tokenizer = tokenizer
         self.device = device
        #预测方法
    def predict(self,inputs:str|list[str]):
        #如果是一条数据，转换成列表处理
        is_str = isinstance(inputs,str)
        if is_str:
            inputs = [inputs]
        #1.预分词，得到字符列表
        tokens_list = [list(input)for input in inputs]
        #2.用分词器id处理
        inputs_tensor =self.tokenizer(
            tokens_list,
            is_split_into_words=True,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        #3.加载到设备
        inputs_tensor={k:v for k,v in inputs_tensor.items()}
        #4.前向传播，推理预测
        with torch.no_grad():
            outputs = self.model(**inputs_tensor)
            logits = outputs.logits
            predictions = logits.argmax(dim=-1)
            #
        #5.将id列表转换为BIO标签,
        final_predictions = []
        for tokens,prediction in zip(tokens_list,predictions):#截取预测输出中，真实长度
            prediction = prediction[1:len(tokens)+1]
            #转化成标签
            final_prediction =[self.model.config.id2label[id.item()] for id in prediction]
            final_predictions.append(final_prediction)
        if is_str:
            return final_predictions[0]
        return final_predictions
    def extract(self,inputs:str|list[str]):
        is_str = isinstance(inputs, str)
        if is_str:
            inputs = [inputs]
        predictions = self.predict(inputs)
        #从当前列表中抽取实体列表
        entities_list = []
        for input ,labels in zip(inputs,predictions):#抽取一个数据样本所有实体标签
            entities=self._extract_entities(list(input),labels)
            entities_list.append(entities)
        if is_str:
            return entities_list[0]
        return entities_list
    def _extract_entities(self,tokens,labels):
        entities =[]
        current_entity =''
        for token,label in zip(tokens,labels):
            #如果标签是B，开始保存新的实体
            if label=='B':
                if current_entity :
                    entities.append(current_entity)
                current_entity=token
            elif label=='I':
                    current_entity+=token
            #如果标签是O
            else:
                if current_entity:#不为空就保存
                    entities.append(current_entity)
                current_entity=''
        if current_entity:
            entities.append(current_entity)
        return entities
    #if is_str:
       # return final_predictions[0]
       # return final_predictions
def predict():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AutoModelForTokenClassification.from_pretrained(str(CHECKPOINT_DIR/NER_DIR/'best_model'))
    tokenizer = AutoTokenizer.from_pretrained(str(CHECKPOINT_DIR/NER_DIR/'best_model'))
    #定义预测器
    predictor = Predictor(model,tokenizer,device)
    text ='麦德龙德国进口双心多维叶黄素护眼营养软胶囊30粒x3盒眼干涩'
    #result = predictor.predict(text)
    #for token,label in zip(text,result):
        #print(token,label)
    #抽取实体
    entities = predictor.extract(text)
    print(entities)
if __name__ == "__main__":
    predict()
