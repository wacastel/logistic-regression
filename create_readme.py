import os

def generate_readme():
    readme_content = r"""# The One with the Logistic Regression

This repository contains a Natural Language Processing (NLP) machine learning pipeline that predicts which character from the TV show *Friends* said a specific quote. 

## 1. The Theory and Math Behind Logistic Regression

Despite its name, Logistic Regression is a **classification** algorithm, not a regression algorithm. It is used to predict the probability that a given input belongs to a certain class.

### The Linear Foundation
Logistic regression builds upon the mechanics of linear regression. It calculates a weighted sum of the input features (in our case, the vectorized words), plus a bias term.

$$ z = \mathbf{w}^T \mathbf{x} + b $$

Where:
* $\mathbf{x}$ is the feature vector (the input data).
* $\mathbf{w}$ is the vector of weights learned by the model.
* $b$ is the bias term (the intercept).
* $z$ is the raw continuous output (the log-odds).

### The Sigmoid Function (Mapping to Probabilities)
Because $z$ can be any number from negative infinity to positive infinity, we need to map it to a probability between 0 and 1. We do this using the **Sigmoid (or Logistic) function**:

$$ \sigma(z) = \frac{1}{1 + e^{-z}} $$

Our predicted probability ($\hat{y}$) for a binary classification is:

$$ \hat{y} = P(y=1 | \mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x} + b) $$

*(Note: For our specific project with 6 characters, this is generalized into Multinomial Logistic Regression using the **Softmax function**, which calculates a probability distribution across all 6 classes so that they sum to 1).*

### Deriving the Cost Function (Log-Loss / Cross-Entropy)
To train the model, we need a way to measure how "wrong" its predictions are. We cannot use Mean Squared Error (MSE) like in linear regression because the sigmoid function makes the loss landscape non-convex (full of local minimums). 

Instead, we use **Binary Cross-Entropy** (or Log-Loss), derived from Maximum Likelihood Estimation (MLE).

The probability of observing the actual label $y$ given our prediction $\hat{y}$ is:

$$ P(y | \mathbf{x}) = \hat{y}^y (1 - \hat{y})^{1 - y} $$

To make the math easier (turning multiplication into addition), we take the natural logarithm. To frame it as a *loss* function (where lower is better), we multiply by -1. This gives us the cost function for a single training example:

$$ L(\hat{y}, y) = - [y \log(\hat{y}) + (1 - y) \log(1 - \hat{y})] $$

For the entire dataset of $m$ examples, the total Cost Function $J(\mathbf{w}, b)$ is the average of all individual losses:

$$ J(\mathbf{w}, b) = -\frac{1}{m} \sum_{i=1}^{m} [y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)})] $$

### Gradient Descent (Learning the Weights)
To minimize this cost function and find the optimal weights ($\mathbf{w}$) and bias ($b$), the algorithm uses **Gradient Descent**. It calculates the partial derivatives (gradients) of the cost function with respect to the weights.

Using the chain rule, the derivative for the weights simplifies elegantly to:

$$ \frac{\partial J}{\partial w_j} = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)}) x_j^{(i)} $$

The algorithm then updates the weights iteratively by taking a step in the opposite direction of the gradient, controlled by a learning rate ($\alpha$):

$$ w_j := w_j - \alpha \frac{\partial J}{\partial w_j} $$

---

## 2. How Scikit-Learn's `LogisticRegression` Works

While the math above describes the raw mechanics, `scikit-learn` implements highly optimized versions of this algorithm under the hood:

1. **Optimization Algorithms (Solvers):** By default, scikit-learn does not use basic gradient descent. It uses advanced solvers like **L-BFGS** (Limited-memory Broyden–Fletcher–Goldfarb–Shanno), which approximate the second derivative (Hessian matrix) to converge on the optimal weights much faster and more reliably.
2. **Multiclass Handling:** Because we have 6 labels, scikit-learn automatically detects this and applies a Multinomial (Cross-Entropy) loss function rather than the binary version, calculating probabilities for all classes simultaneously.
3. **Regularization:** By default, scikit-learn applies **L2 Regularization (Ridge)**. This adds a penalty term to the cost function ($\frac{\lambda}{2} \|\mathbf{w}\|^2$) to prevent the weights from getting too large. This is a built-in defense against overfitting!

---

## 3. How the TF-IDF Vectorizer Works

Machine learning models only understand numbers, not text. We used a `TfidfVectorizer` to convert our Friends quotes into a mathematical matrix. 

**TF-IDF** stands for **Term Frequency - Inverse Document Frequency**. It scores words based on two components:

### 1. Term Frequency (TF)
How often does a word appear in a specific quote? 
$$ TF(t, d) = \frac{\text{Count of term } t \text{ in document } d}{\text{Total words in document } d} $$
*If a quote is "Pivot, pivot, pivot!", the word "pivot" has a very high TF.*

### 2. Inverse Document Frequency (IDF)
How common or rare is the word across the *entire* TV show? 
$$ IDF(t) = \log\left(\frac{N}{df(t)}\right) $$
Where $N$ is the total number of quotes, and $df(t)$ is the number of quotes containing the term $t$.
*Common words like "the" or "and" appear in almost every quote, so their IDF score approaches 0. Rare words like "dinosaur" or "transponster" get a high IDF score.*

### The Final Score
$$ \text{TF-IDF} = TF \times IDF $$

By multiplying them together, the vectorizer mathematically highlights the unique "signature" words of a character while ignoring the generic English vocabulary. This creates the exact feature matrix $\mathbf{X}$ that our Logistic Regression model uses to make its predictions.
"""

    with open("README.md", "w", encoding="utf-8") as file:
        file.write(readme_content)
        
    print("Successfully created README.md!")
    print("You can now open the file to preview the Markdown and mathematical formulas.")

if __name__ == "__main__":
    generate_readme()
