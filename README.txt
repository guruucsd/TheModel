Before you start...

0) Make sure your directory is set up the way I had it. This will help the
   import.py class work.
 - I have it as: /Users/Davis/Desktop/theModel
 - theModel has 2 folders: code, data
 - code is where all the python scripts go. data holds all the datasets you
   want to use for your demo.
 - data will contain another folder: dummy

1) Make sure all you image data is labeled correctly.
 - Make sure the first character in your image name is a number that corresponds
   to the set it is in.
 - For example, 1face.jpg, 2face,jpg, 3face.jpg will correspond to faces in
   various categories. 1-happy, 2-sad, 3-surprised, etc...

2) Make sure to edit the ANNClass.py file to make sure your targets are set up
   correctly. Currently, I only have 2 output units. Edit the document to
   reflect the amount of output units/ how the outputs are organized to suit
   your own needs. For facial affect recognition, it would make sense to have
   7 outputs, one for each emotion.
  - Finally, remember to edit the code that transforms your labels to targets in
    the ANNClass.py file.

3) note that filtering is slower than matlab... Also, PCA is not yet
   implemented. Additionally, you may want to add code to separate the training set
   with the testing/validation set and code to tell you the current test
   set/validation set error percentages.
