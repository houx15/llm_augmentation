## LLM Augmentation
This project tries to augment reasonable social survey data using LLMs.

### Related Literature
Few literature has focused on the augmentation of social survey data. This project gets the insight from a missing-values imputation work, IPM.
In this paper, to improve the performance of an imputation model fine-tuned based on Bert, the researchers use neighbor-based imputation methods to generate a set of candidates first, and then fine-tune a model to predict whether the candidate is the correct prediction or not.

### Work Flow
![Work Flow](image.png)

E.G.
Model Input
>>>>>>>>> 

### Performance

### To-Dos
[ ] Use regression task instead of binary and consider the distance of one augmented data from the original one
