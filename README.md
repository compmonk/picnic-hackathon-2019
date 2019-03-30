# Picnic Hackathon 2019
 
Image classification (https://picnic.devpost.com)

## Setup
```
git clone https://github.com/compmonk/picnic-hackathon-2019.git
cd picnic-hackathon-2019
virtualenv venv
virtualenv -p /usr/bin/python3 venv
source venv/bin/activate
pip install -r requirements.txt
mkdir data
unzip 'The Picnic Hackathon 2019.zip' 'The Picnic Hackathon 2019/*' -d data 
```

## To run
```
source venv/bin/activate
# to generate samples of training and test data
python sampler.py

# to generate models
python cnn/model_generator.py

# to classify
python cnn/predict.py
```

Download data set from https://drive.google.com/open?id=1XSoOCPpndRCUIzz2LyRH0y01q35J7mgC
