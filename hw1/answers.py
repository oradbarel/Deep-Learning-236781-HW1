r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers

part1_q1 = r"""
**Your answer:**

1.1:
False.
The in-sample error assesses the accuracy of a function in fitting a particular training dataset.
Conversely, the test set is used to "mimic" unknown samples that are not included in the training process and hence can't be used to determine the in-sample error.
Instead, the error measured on the test set is referred to as the out-sample error.

1.2:
False.
As an illustration, having a small test set would make it difficult to accurately assess the generalization error.
Similarly, an insufficiently large training set may cause the model to overfit on a specific set of samples.

1.3:
True.
It's crucial that the test set remains independent of the training process to accurately simulate unknown data.
As part of the training process, cross-validation is used to optimize hyperparameters, and as such, the test set should not be incorporated.
During cross-validation, instead of using the test set, we create a validation subset from the training set and train the model with the remaining data.
This process allows us to validate the results.

1.4:
True.
Validation sets provide valuable insights into a model's generalization ability as they show the performance of a model,
using specific hyperparameters, on samples that were not used during the training process.
Therefore, the validation set results indeed serve as an approximation for the model's generalization error.

"""

part1_q2 = r"""
**Your answer:**

The approach described above is not justified.
Selecting the optimal hyperparameters is an integral aspect of the training stage, and as such, the test set should not be incorporated into this process.
Instead, cross-validation should be conducted within the training set to assess the model's performance under specific hyperparameters.
This preserves the true purpose of the test set, enabling us to evaluate the performance of the tuned model on previously unseen data.
If the test set is used during the training stage, it can compromise our ability to calculate an unbiased generalization error.

"""

# ==============
# Part 2 answers

part2_q1 = r"""
**Your answer:**

Increasing the value of k does not always lead to improved generalization of unseen data, as it can lead to over simplified decision boundaries that may not suit our dataset.
For example, if k equals the number of samples used to train the model, the outcome would be a model with identical predictions for all samples.
However, increasing k can minimize the influence of outliers and noise, which can ultimately improve the generalization error.
Therefore, we should increase k up to a certain threshold to mitigate the impact of outliers and noise, but not to a level that produces excessively blunt decision boundaries,
and thus, reduces the model's ability to classify accurately.
"""

part2_q2 = r"""
**Your answer:**

2.1:
If we train the model on the entire training dataset and select the model with the highest accuracy on the train set,
we won't be able to assess the performance of the selected model on new data.
This may result in choosing an overfitting model that only performs well on the training data.
However, if we use cross-validation, we can obtain results for each model on how well it performs on data that was not used in the training process.
This approach reduces the likelihood of selecting an overfitting model and helps us to better validate the performance of the selected model on new, unseen data.


2.2:
If we use the test set to select a model, we risk losing the ability to accurately measure the generalization error at the end of the training process,
as the final model will be influenced by the test set.
This can lead to selecting a model with poor generalization performance.
However, by using cross-validation, we can choose hyperparameters based on how well the model performs on data that was not used in the training process,
without compromising the ability to measure the generalization error accurately.
"""

# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**Your answer:**

The selection of $\Delta > 0$ is indeed arbitrary for the SVM loss $L(\mat{W})$ as defined above. let us assume we scale $\Delta$ by a scalar $\alpha>0$.
then $W$ can be scaled by $\alpha$ as well:
$$L_{i}( \alpha \mat{W}) =  \sum_{j \neq y_i} \max\left(0, \alpha \Delta+ \alpha \vec{w_j}^T \vec{x_i} - \alpha \vec{w_{y_i}}^T \vec{x_i}\right) = \alpha L_{i}(\mat{W})$$
$$L(\alpha \mat{W}) =
\frac{1}{N} \sum_{i=1}^{N} L_{i}(\alpha \mat{W})
+
\frac{\lambda}{2} \lVert{\alpha \mat{W}}\rVert_2^2 = 
\frac{1}{N} \sum_{i=1}^{N}\alpha L_{i}( \mat{W})
+
\frac{\alpha \lambda}{2} \lVert{\mat{W}}\rVert_2^2 = 
\alpha  L(\mat{W})$$
And, of course, scaling the Loss does not matter since it doesn't affect the gradients, thus the training would yield the same optimum.

"""

part3_q2 = r"""
**Your answer:**

1. Our interpretation of the linear model's learning can be described as: how likely is each pixel in the image, to fit each class (in terms of class's properties).
For example, for the class of the digit "1": for images with shapes of a straight vertical line - the model would grant a high 1-digit-class score, and a low such score for images with different shapes.
We notice that for digits that look pretty similar (with high correlations between them, like 6 and 5, or 7 and 9, or 4 and 9), the model might not predict well and make an error.

2. The idea that the linear model learns from all examples and generates a weight vector that can be viewed as a close neighbor to class examples is similar to the KNN approach.
The linear model can be thought of as a variation of the 1NN method, with each class weight vector serving as a neighbor.
However, unlike the KNN model, which is influenced only by K of the closest neighbors, classification in the linear model is impacted by all of the training examples.
"""

part3_q3 = r"""
**Your answer:**

1. Based on the graph of the training set loss, the learning rate we chose is good.
In case the learning rate was too low: the graph would look more linear - more slowly converging to the minimum.
In case the learning rate was too high: the graph would drop down faster but wouldn't smoothly converge at the end, instead oscillate near the minimum.

2. Based on the graph of the training and test set accuracy, the model is slightly overfitted to the training set.
The indication to the overfit behavior is the training set accuracy being higher than the test set accuracy. it is only slight since the difference between them is very small.
Also, the model is not underfitted to the training set since the accuracies are very high (more than 90%).
"""

# ==============

# ==============
# Part 4 answers

part4_q1 = r"""
**Your answer:**

The ideal pattern to see in a residual plot is an even horizontal mass being very close to 0 value, this would indicate a good prediction (small error). The closet to zero the smallest the error is.
Observing our residual plot:
First for the top-5 features we got a graph with a certain curve that goes farther from zero, which indicates error (model doesn't learn well).
After we added non-linearity, it improved and the graph seems more evenly spread inside the margins, closer to zero.
After cross-validation we got the final graph which is close to ideal as we described above.
In terms of overfitting - we wish the behavior of both train and test sets to be similar which indeed happens, this indicates the model generalizes well.
"""

part4_q2 = r"""
**Your answer:**

1. After adding non-linear features it is still a linear regression model. the non-linear features simply map the data to a higher dimension, but the desicion boundry is still linear (upon the mapped features).

2. Yes, in order to do so we need to find a feature mapping (tranformation) to a space in which the function behaves linearly, this way with these certain non-linear features we can fit (and might overfit) any non-linear function.

3. Adding non-linear features, performing feature mapping, means the model will be learning in the new features' space. In it, as we explained before, the model is still a linear regressor and would yield a hyperplane in the new space.
Although, the decision boundry learned in the new space is not linear in the original space (it is not a hyperplane in the original space), meaning it is not a hyperplane, and it is more complex.
"""

part4_q3 = r"""
**Your answer:**

1. using np.logspace gives us the advantage of better exploration - with it we can examine both very small values and bigger values without having a too big amount of values to check.
If we use np.linspace we might not reach small enough values (abs) or big enough values, or in order to not miss any - we would need to explore over a too big amount of values.

2. The model was fitted 180 times.
For each combination of lambda and degree value we fitted the model K times (number of folds). 3 degree values and 20 lambda values define 60 hypermeter combinations, each fitted 3 times (3\*20\*3=180).
"""

# ==============
