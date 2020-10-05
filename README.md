# Vision foodvisor

Home assignment - Tomato allergies

## Context

In the context of meal logging, there is a fluctuating asymmetry between task evaluation (user expectation) and data qualification for training (result of labeling), that require specific care. Additionally, in order to have proper class separation, training sometimes requires a superior granularity compared to evaluation's. 

The research team of Foodvisor is in charge of creating the detection algorithm used for meal logging with the mobile app. Opening access to the app in a new region of the world usually brings about new user expectations.

Source assignment : https://github.com/Foodvisor/home-assignment

## Assignement #1 and #2

We strongly recommand to use jupyter notebook in order to understand and read the step by step solution written in 
the TomatoFinder.ipynb file

Please install all required packages with `pip install -r requirements.txt`
Please insert the data in the project directory as follow :

```bash
.
├── data
│   ├── assignment_imgs
│   │   ├── img1.ext
│   │   ├── img2.ext
│   │   ├── img3.ext
│   │   └── img4.ext
│   ├── img_annotations.json
│   └── label_mapping.csv
│
├── output
│   └─ vgg_model1.pth
│   
├── requirements.txt
├── .gitignore
├── TomatoFinder.ipynb
└── README.md
```

This repository present the results of both exercices : 

First we successfully build a binary classifier that achieve more that **90%** accuracy and recall.

The function has_tomatoes() is built ! Try it with the command `python detect.py -i image_path.ext `

Example `python detect.py -i data/assignment_imgs/1bc35c0a1a67cb25f99bbfde2192dc28.jpeg ` 

Should print the following :
`
There is some tomatoes !! Be careful
` 

And `python detect.py -i data/assignment_imgs/1b54ec8350f5dac91cf5bcec92bce32b.jpeg ` 

Should print the following :
`
No tomatoes detected here
` 

Then we implemented a GradCam to get a localization module

## Credits 

[paper Grad-CAM](https://arxiv.org/pdf/1512.04150.pdf)