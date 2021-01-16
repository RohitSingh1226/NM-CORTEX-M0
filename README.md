# NM-CORTEX-M0

Naive bayes make stringent assumptions about the data, that they wont perform well with complex data,it is an extremely robust method for training and prediction, and is very accurate and user friendly because it has very few parameters to tweak.
We will use the simple example to show the working on the gaussian Naive bayes.
We are given the height weight and foot size of a person and we have to predict if its a male or a female. We will use a python script, to get the gaussian means, variances and class probabilities, that will later be computed by our cortex M0+ microcontroller. The code has been optimized to exibhit minimal power consumption.
Let us start with the python script to calculate the thetha,sigma and the probability. We will use scikit library to import a dataset. 
The vectors are generated and the classes are defined for training the code. The d
Once the data is fed in the python script 
We print the values of the gaussian mean and variance 


Prior probab==0.5 ,0.5

Conditional probabilities = 


These will be fed directly in the M0 cortex code 

Thetha=mean sigma =variance 

Then we run the keil file to get the output
###### RAN INTO A FEW ERRORS NOT COMPLETE BEFORE SATURDAY###################









According to the function of arm_gaussian_naive_bayes_predict_f32, it puts each probability to the buffer and returns the index of the highest probability value.
But in the example code of arm_bayes_example_f32.c, it calculates the max value in the result buffer even though arm_gaussian_naive_bayes_predict_f32 returns it.
  arm_gaussian_naive_bayes_predict_f32(&S, in, result);

  arm_max_f32(result, NB_OF_CLASSES, &maxProba, &index);
I think this is a bit redundant. So this is clearer.
  index = arm_gaussian_naive_bayes_predict_f32(&S, in, result);

  maxProba = result[index];

In the example I wanted to extract the probability in addition to the index.
But it does not make sense to re-compute the max for this. Hence the above is used.
