import os
import sys
import json
import urllib.request
import re
import random
import pandas as pd
import numpy as np
import pickle
import torch
import collections

from datasets import Dataset, load_dataset
from transformers import AutoModelForMaskedLM
from transformers import AutoTokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import default_data_collator
from transformers import TrainingArguments
from transformers import Trainer
from transformers import pipeline

from nltk.featstruct import RangeFeature
from konlpy.tag import Okt

class data_augmentation:

  def __init__(self, mode=None, client_id=None, client_secret=None, target_lang ='en', prob=1, num_per_label=None):

    """파라미터에 관한 설명
        1. mode : 어떤 증강을 시킬것인지 선택을 하는 파라미터입니다.
         - 'back translation' : 역번역을 진행합니다.
         - 'jamo split' : 자음과 모음을 분리합니다.
         - 'vowel change' : 특정 모음을 다른 모음으로 대체합니다.
         - 'add dot' : 문장 중간중간에 온점을 추가합니다.
         - 'kor2eng' : 특정 문자를 알파벳으로 변형합니다.
         - 'yamin' : 임의의 단어를 야민정음으로 변경합니다.
         - 'eda' : eda 증강을 진행합니다.
         - 'bert' : bert 모델을 활용한 데이터 증강을 진행합니다.
         - 'gpt' : gpt 모델을 활용한 데이터 증강을 진행합니다.
       2. client_id : back translation 사용시 필요한 개인정보입니다. 네이버 파파고 번역 api를 개인적으로 등록해서 받아야 합니다.
       3. client_secret : back translation 사용시 필요한 개인정보입니다. 네이버 파파고 번역 api를 개인적으로 등록해서 받아야 합니다.
       4. target_lang : back translation 사용시 번역해줄 다른 언어입니다.
       5. prob : 노이즈 생성을 얼마나 할지 확률입니다.
       6. num_per_label : 샘플링할 시 라벨당 모을 텍스트의 수입니다.
    """

    self.mode = mode
    self.num_per_label = num_per_label

    self.client_id = client_id
    self.client_secret = client_secret
    self.target_lang = target_lang

    self.consonant = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
    self.vowel = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ','ㅢ', 'ㅣ']
    self.final_consonant = [' ', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ','ㅂ','ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
    self.pairs = {'ㅏ': 'ㅑ', 'ㅑ': 'ㅏ', 'ㅓ': 'ㅕ', 'ㅕ': 'ㅓ', 'ㅗ': 'ㅛ', 'ㅛ': 'ㅗ', 'ㅜ': 'ㅠ', 'ㅠ': 'ㅜ', }
    self.oral_consonant = ['ㄱ', 'ㄷ', 'ㄹ', 'ㅂ', 'ㅅ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅎ']
    self.nasal_consonant = ['ㅁ', 'ㄴ', 'ㅇ']
    self.liquid_consonant = ['ㄹ']
    self.ke_cons = {'ㅌ': 'E', 'ㄱ': '7', 'ㄴ': 'L', 'ㅇ': 'O'}
    self.ke_vowel = {'ㅏ': 'r', 'ㅣ': 'l', 'ㅐ': 'H'}
    self.ya_min_jung_um = {'ㄷㅐ': 'ㅁㅓ', 'ㅁㅕ': 'ㄸㅣ', 'ㄱㅟ':'ㅋㅓ', 'ㅍㅏ':'ㄱㅘ', 'ㅍㅣ':'ㄲㅢ', 'ㅇㅠ ':'ㅇㅡㄲ', 'ㄱㅜㅅ':'ㄱㅡㅅ'}
    self.exceptions = ['ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅗ']
    self.prob=prob

    self.mask_filler=None

  def augmentation(self, text):

    if self.mode == 'back translation': 
      translated_data = self.korean_to_english(text)      
      aug_data = self.english_to_korean(translated_data)
      return aug_data

    if self.mode == 'jamo split':
      return self.splitting_noise(text, prob=self.prob)

    elif self.mode == 'vowel change':
      return self.vowel_noise(text, prob=self.prob)

    elif self.mode == 'add dot':
      return self.add_dot(text, prob=self.prob)

    elif self.mode == 'kor2eng':
      return self.replace_kor_eng(text, prob=self.prob)

    elif self.mode == 'yamin':
      return self.yamin(text, prob=self.prob)

    elif self.mode == 'eda':
      return self.EDA(text)

    elif self.mode == 'gpt':
      return None


  # csv 파일을 불러와서 label과 text를 읽어주는 함수입니다.
  # root : 파일 경로에 해당하는 변수.
  # label_column : 데이터프레임에서 라벨에 해당하는 인덱스.(ex : 웰니스데이터셋의 intent 인덱스인 2.)
  # text_column : 데이터프레임에서 텍스트에 해당하는 인덱스.(ex : 웰니스데이터셋의 uteerance(2차)의 인덱스인 6.)
  def load_csv(self, root, label_column, text_column):
    df = pd.read_csv(root, sep=',')
    label = np.array(df.iloc[:,[label_column]])
    text = np.array(df.iloc[:, [text_column]])
    return label, text


  #샘플링을 진행하는 함수입니다.
  def make_sample(self, root, label_column, text_column):
    label, text = self.load_csv(root, label_column, text_column)
    if self.num_per_label==None:
      return label, text
    else:
      return None

  def generation(self, root, label_column, text_column):
    if self.mode=='bert':
      return self.bert_fit(root, label_column, text_column)
    else:
      label, text = self.make_sample(root, label_column, text_column)
      augmentated_text=[]
      for t in text:
        augmentated_text.append(self.augmentation(str(t[0])))
      return augmentated_text

  def korean_to_english(self, data):
    client_id = self.client_id
    client_secret = self.client_secret
    encText = urllib.parse.quote(data)
    data = "source=ko&target=en&text=" + encText
    url = "https://openapi.naver.com/v1/papago/n2mt"
    request = urllib.request.Request(url)
    request.add_header("X-Naver-Client-Id",client_id)
    request.add_header("X-Naver-Client-Secret",client_secret)
    response = urllib.request.urlopen(request, data=data.encode("utf-8"))
    rescode = response.getcode()
    if(rescode==200):
        response_body = response.read()
        return(json.loads(response_body.decode('utf-8'))['message']['result']['translatedText'])
    else:
        print("Error Code:" + rescode)

  def english_to_korean(self, data):
    client_id = self.client_id
    client_secret = self.client_secret
    kocText = urllib.parse.quote(data)
    data = "source=en&target=ko&text=" + kocText
    url = "https://openapi.naver.com/v1/papago/n2mt"
    request = urllib.request.Request(url)
    request.add_header("X-Naver-Client-Id",client_id)
    request.add_header("X-Naver-Client-Secret",client_secret)
    response = urllib.request.urlopen(request, data=data.encode("utf-8"))
    rescode = response.getcode()
    if(rescode==200):
        response_body = response.read()
        return(json.loads(response_body.decode('utf-8'))['message']['result']['translatedText'])
    else:
        print("Error Code:" + rescode)


  def load_pairs(self, path):
      with open(path, 'r', encoding='utf-8') as r:
          contents = [l.split('\t') for l in r.read().split('\n')]
      dictionary = {k: v for k, v in contents if k != v and k.strip() != ''}
      return dictionary


  def jamo_split(self, char):
      base = ord(char) - ord('가')
      c = base // 588
      v = (base - 588 * c) // 28
      f_c = base - 588 * c - 28 * v
      return [self.consonant[c], self.vowel[v], self.final_consonant[f_c]]


  def jamo_merge(self, jamo_list):
      if jamo_list[1:] == ['', '']:
          return jamo_list[0]
      c, v, f_c = [_list.index(j) for _list, j in zip([self.consonant, self.vowel, self.final_consonant], jamo_list)]
      return chr(f_c + 588 * c + 28 * v + ord('가'))


  def splitting_noise(self, content, prob=0.1):
      condition = lambda xlist: ((xlist[-1] == ' ') and (xlist[-2] not in self.exceptions))

      output = [self.jamo_split(ch) if re.match('[가-힣]', ch) else [ch, '', ''] for ch in content]

      output = [''.join(out).strip() if condition(out) and (random.random() < prob) else content[i] for i, out in
                enumerate(output)]

      return ''.join(output)


  def vowel_noise(self, content, prob=0.1):
      output = [self.jamo_split(ch) if re.match('[가-힣]', ch) else [ch, '', ''] for ch in content]
      condition = lambda xlist: ((xlist[-1] == ' ') and (xlist[-2] in self.pairs))
      output = [
          self.jamo_merge([out[0], self.pairs[out[1]], out[2]]) if condition(out) and (random.random() < prob) else
          content[i] for i, out in
          enumerate(output)]
      return ''.join(output)


  def add_dot(self, contents, prob=0.3):
      indexes = random.sample(range(len(contents)), int(len(contents)*prob))
      contents = ''.join([c+'.' if i in indexes else c for i,c in enumerate(contents)])
      return contents


  def replace_kor_eng(self, content, prob=0.1):
      condition = lambda xlist: (xlist[2] == ' ' and (xlist[1] in self.ke_vowel or xlist[0] in self.ke_cons) and xlist[1] not in self.exceptions)
      mapping = lambda dic,q: dic[q] if q in dic else q
      output = [self.jamo_split(ch) if re.match('[가-힣]', ch) else [ch, '', ''] for ch in content]

      output = [''.join([mapping(self.ke_cons,out[0]), mapping(self.ke_vowel, out[1]), out[2]]).strip() if condition(out) and (random.random() < prob) else content[i] for i, out in
                enumerate(output)]
      return ''.join(output)


  def ya(self, charlist):
      out = ''.join(charlist)
      for k,v in self.ya_min_jung_um.items():
          if k in out:
              out = out.replace(k,v)
              break
      return list(out)


  def yamin(self, content, prob=0.1):
      condition = lambda xlist: xlist[-1] != ''
      output = [self.jamo_split(ch) if re.match('[가-힣]', ch) else [ch, '', ''] for ch in content]
      output = [self.jamo_merge(self.ya(out)) if (random.random() < self.prob) and condition(out) else content[i]
                for i, out in enumerate(output)]
      return ''.join(output)


  def get_only_hangul(self,line):
    parseText= re.compile('/ ^[ㄱ-ㅎㅏ-ㅣ가-힣]*$/').sub('',line)

    return parseText 


  def random_deletion(self, words, p):
    if len(words) == 1:
      return words

    new_words = []
    for word in words:
      r = random.uniform(0, 1)
      if r > p:
        new_words.append(word)

    if len(new_words) == 0:
      rand_int = random.randint(0, len(words)-1)
      return [words[rand_int]]

    return new_words


  def swap_word(self, new_words):
    random_idx_1 = random.randint(0, len(new_words)-1)
    random_idx_2 = random_idx_1
    counter = 0

    while random_idx_2 == random_idx_1:
      random_idx_2 = random.randint(0, len(new_words)-1)
      counter += 1
      if counter > 3:
        return new_words

    new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1]
    return new_words


  def random_swap(self, words, n):
    new_words = words.copy()
    for _ in range(n):
      new_words = self.swap_word(new_words)

    return new_words

  def EDA(self, sentence, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, p_rd=0.1):
    sentence = self.get_only_hangul(sentence)
    words = sentence.split(' ')
    words = [word for word in words if word is not ""]
    num_words = len(words)

    n_rs = max(1, int(alpha_rs*num_words))

    # rs
    a_words = self.random_swap(words, n_rs)
    augmentated_text = " ".join(a_words)

    words = augmentated_text.split(' ')
    words = [word for word in words if word is not ""]
    
    # rd
    a_words = self.random_deletion(words, p_rd)
    augmentated_text = " ".join(a_words)


    return augmentated_text


  # bert 활용 증강 코드입니다.
  def bert_fit(self, root, label_column, text_column):
    
    df = pd.read_csv(root, sep=',')	# csv 파일을 읽습니다.
    df =df.iloc[:,[2,6]]			# 텍스트와 라벨 열만 읽습니다.
    df.columns = ['label', 'text']		# 레이블과 텍스트로 열 이름을 다시 지정해줍니다.

    concated = []				
    for i in range(len(df)):	
      concated.append(str(df['label'][i]) + ' ' + str(df['text'][i]))	#라벨이 포함된 텍스트로 전처리를 해줍니다.
    pre_df = pd.DataFrame({'data' : concated})	#전처리된 데이터로 데이터프레임을 만듭니다.

    dataset = Dataset.from_pandas(pre_df, split="validtion")
    dataset = dataset.train_test_split(test_size=0.2, shuffle=True) #검증데이터와 훈련데이터로 나눠줍니다.

    #klue/bert 모델, 토크나이저 불러오기
    model_checkpoint = "klue/bert-base"
    model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    #tokenization 진행

    #text 컬럼 삭제
    def tokenize_function(examples):
        result = tokenizer(examples["data"])
        print(examples["data"])
        if tokenizer.is_fast:
            result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
        return result


    # Use batched=True to activate fast multithreading!
    tokenized_datasets = dataset.map(
        tokenize_function, batched = True, remove_columns=["data"]
          )

    chunk_size = 128
    def group_texts(examples):
        # Concatenate all texts
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        # Compute length of concatenated texts
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the last chunk if it's smaller than chunk_size
        total_length = (total_length // chunk_size) * chunk_size
        # Split by chunks of max_len
        result = {
            k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
            for k, t in concatenated_examples.items()
        }
        # Create a new labels column
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = tokenized_datasets.map(group_texts, batched=True)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

    wwm_probability = 0.2

    def whole_word_masking_data_collator(features):
        for feature in features:
            word_ids = feature.pop("word_ids")

            # Create a map between words and corresponding token indices
            mapping = collections.defaultdict(list)
            current_word_index = -1
            current_word = None
            for idx, word_id in enumerate(word_ids):
                if word_id is not None:
                    if word_id != current_word:
                        current_word = word_id
                        current_word_index += 1
                    mapping[current_word_index].append(idx)

            # Randomly mask words
            mask = np.random.binomial(1, wwm_probability, (len(mapping),))
            input_ids = feature["input_ids"]
            labels = feature["labels"]
            new_labels = [-100] * len(labels)
            for word_id in np.where(mask)[0]:
                word_id = word_id.item()
                for idx in mapping[word_id]:
                    new_labels[idx] = labels[idx]
                    input_ids[idx] = tokenizer.mask_token_id

        return default_data_collator(features)

    batch_size = 8
    # Show the training loss with every epoch
    logging_steps = len(lm_datasets["train"]) // batch_size
    model_name = model_checkpoint.split("/")[-1]

    training_args = TrainingArguments(
        output_dir=f"{model_name}-low_resource-dataset",
        evaluation_strategy="epoch",
        learning_rate=4e-5,
        weight_decay=0.01,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        fp16=True,
        logging_steps=logging_steps,
        num_train_epochs=10,
    )

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_datasets["train"],
    eval_dataset=lm_datasets["test"],
    data_collator = data_collator,
    )

    trainer.train()
    trainer.evaluate()
    trainer.save_model("./bert")
    model = AutoModelForMaskedLM.from_pretrained("./bert")


    self.mask_filler = pipeline(
    "fill-mask", model=model, 
    tokenizer=tokenizer
    )

    df = df.dropna(axis=0)  # 결측치 제거
    original = list(df['text'])

    okt = Okt()
    for i in range(len(original)):
      tokenized_ex = okt.morphs(original[i])
      len_tokens = len(tokenized_ex)
      if len_tokens == 1:
        masked_idx=0
      else:
        masked_idx = random.randint(0,len_tokens-1)
      tokenized_ex[masked_idx] = '[MASK]'
      masked_ex = ""
      for x in tokenized_ex:
        masked_ex+=" " + x
      original[i]=masked_ex
    print("1")
    created_text=[]
    for text in original:
      preds = self.mask_filler(text)
      new_text = preds[0]['sequence']
      created_text.append(new_text)

    return created_text