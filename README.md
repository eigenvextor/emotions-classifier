# emotions-classifier

[GoEmotions](https://paperswithcode.com/dataset/goemotions) (a multi-label classification dataset) was used for this project. After some preprocessing & criterias, only 53,951 of the original 58,011 instances are used to reduce noise. Check [data](src/data.py) to know how the data was prepared and [visuals](img) for some visuals of the dataset.

BERT model was used and trained for 3 epochs. It took ~7 hours for training it on my local machine. 

## To train the model locally

```bash
git clone https://github.com/eigenvextor/emotions-classifier.git
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cd src
python3 train.py
```

## To run the streamlit app after training
```bash
streamlit run app.py 
```