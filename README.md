## Unsupervised Cross-subject Adaptation for Predicting Human Locomotion Intent
This is the implementation of Unsupervised Cross-subject Adaptation for Predicting Human Locomotion Intent in Pytorch.

## Getting Started
### Installation
```
pip install -r requirements.txt
```

### Download dataset
If you can use google drive, you don't need to download the data manually and just run the code shown below.

If you cannot use google drive, you need to download the dataset and checkpoint from the link below:

```
https://alumniubcca-my.sharepoint.com/:f:/g/personal/kuangen_zhang_alumni_ubc_ca/EmYydTnluklBn17qVXnSIWoBvBq0arhyATCaVlYXVs4PhA?e=evB0s7
```

### Test
```
python code/main_MCD.py --eval_only True
```

### Train
```
python code/main_MCD.py --eval_only False
```


