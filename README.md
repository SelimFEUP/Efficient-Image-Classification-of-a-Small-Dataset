# Efficient-Sound-Classification-of-a-Small-Dataset
This repository presents a sound classification model of a small dataset ESC-50 using a pre-trained Yamnet model. Tha dataset only has 2000 samples and hence, conventional DL-based models can not learn efficiently. Data augmentation techniques can be used to address the problem, however, from my perspective it will always carry some bias. Hence, utilizing a state-of-the-art pre-trained model is the most suitable solution.
# Dataset
Download the ESC-50 dataset from https://github.com/karolpiczak/ESC-50 and extract it in ./data folder.
# Usage
To train and evaluate the model, from the terminal run 
```bash 
python3 main.py
```
![image](https://github.com/user-attachments/assets/04423551-eeac-4894-b130-2bb70c3eb3cf)

