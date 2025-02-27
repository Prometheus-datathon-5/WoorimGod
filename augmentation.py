import pandas as pd
from sklearn.model_selection import train_test_split
from help_augmentation import data_augmentation
import random


def augmentation_text(train_dataset_name):
    train_data = pd.read_csv(train_dataset_name, sep=",")
    train_data.drop('id', axis=1, inplace=True)
    train_data.rename(columns = {"paragraph":"news"}, inplace=True)
    
    
    train, test = train_test_split(train_data, test_size=0.2, random_state=42)
    
    augmentor = data_augmentation(mode='eda')
    augmentor2 = data_augmentation(mode='add dot')
    augmentor3 = data_augmentation(mode="yamin")
    augmentor4 = data_augmentation(mode="vowel change")
    augmentor5 = data_augmentation(mode="jamo split")

    new_aug_list = []

    for i in range(len(train)):
        new_paragraph = augmentor.augmentation(train.iloc[i][0])
        new_paragraph2 = augmentor2.augmentation(train.iloc[i][0])
        new_paragraph3 = augmentor3.augmentation(train.iloc[i][0])
        new_paragraph4 = augmentor4.augmentation(train.iloc[i][0])
        new_paragraph5 = augmentor5.augmentation(train.iloc[i][0])

        new_aug_list.append((new_paragraph, train.iloc[i][1]))
        new_aug_list.append((new_paragraph2, train.iloc[i][1]))
        new_aug_list.append((new_paragraph3, train.iloc[i][1]))
        new_aug_list.append((new_paragraph4, train.iloc[i][1]))
        new_aug_list.append((new_paragraph5, train.iloc[i][1]))
    
    for i in range(len(new_aug_list)):
        num = random.randint(0, len(train) - 1)
        train.loc[num + 0.5] = [new_aug_list[i][0], new_aug_list[i][1]]
  
    train = train.sort_index().reset_index(drop=True)
    return train, test


train, val = augmentation_text("train.csv")

train.to_csv("train.tsv", sep="\t")
val.to_csv("val.tsv", sep="\t")


