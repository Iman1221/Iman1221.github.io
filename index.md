Last Updated on October 11, 2019
see my [face recognition using webcam project in GitHub](https://https://github.com/Iman1221/Face-gender-race-recognition-with-webcam) for training and testing a face recogniton system.

In this tutorial, we’ll see how to create and launch a face recognition algorithm in Python. We’ll also train a model for face recognition using SVM classifier and Arcface loss.

<img src="https://github.com/Iman1221/Iman1221.github.io/blob/docs/images.jpeg?raw=true"
height="300">

### Fcae recognition
Assume you have some image of known people, and you want to identify these people.
If you want to create a roubust face recognition system, you need to collect as many face images of each person as you can. 
For example, assume you want to identify 5 people as in bellow:

<img src="https://github.com/Iman1221/Iman1221.github.io/blob/docs/Screenshot%20from%202019-10-14%2020-43-07.png?raw=true"
height="150">

I have downloaded three images of each person from google, and put them in the following folders.

In the second step, you should detect faces of every person and put them in another directory. I used MTCNN face detector. The code is in my GitHub page and you can use it like:


```markdown
python align_training_images.py
```
## Creating the classfier

After detecting faces, its time to extract features. Because currently one of the best face recognition algorithms is known as arcface, we use it for face recognition. 
First download the arcface pre-trained model from [here](https://github.com/deepinsight/insightface/wiki/Model-Zoo) and the use train.py from my GitHub repository:

```markdown
python train.py
```

At the end, you have a file named classifier.pkl in directory named trained_classifier.
