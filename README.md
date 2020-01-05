- Select randomly 3 subsets of 10 classes and 2 subsets of 30 classes [Shyam : COMPLETE]
- Report the average classification accuracy over the 5 splits above [Kunal : COMPLETE]
- i) Plot the confusion matrix for one of the 3 random splits of 10 classes [Kunal : COMPLETE]
  ii) report the classification accuracy of each class individually
  iii) plot the confusion matrix for the tasks as well (Question : Meaning confusion matrix for all classes from step ii above?)
- Compute classification accuracies with training and testing on all the 101 classes
	- Report top 5 classes with highest classification accuracy and 5 worst performing classes
	- Explain our results (include in presentation)
- Using responses from final fully connected layer as features train a linear SVM to perform classification (maybe using lib-SVM package)
	- Compare the classication accuracy on the features obtained from using off-the-shelf VGG-16 and ResNet-34.
- Starting with the base VGG-16, fine tune the network to classify the same 10 and 30 splits from the previous part. 
	- Also fine tune the network to classify all the 101 classes.
	- Repeat the same experiment with ResNet-34.
- Incorporate label smoothing regularization into your fine-tuning scheme for both VGG and ResNet and apply to the split considering all 101 classes. 
	- Do the classification results get better or worse? (include in presentation)
- Presentation:
	- motivation and objectives (1-2 slides)
	- short overview of theoretical basis and network architecture (1-2 slides)
	- our experiences : what worked what didn't?
	- experimental results (include the result mentioned in the non-presentation sections)
	- other interesting insights
	- Read the papers and include them in presentation
