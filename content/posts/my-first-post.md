---
title: "Introduction to Feature Attributes in Machine Learning-I"
subtitle: "Marvel at the Extravaganza of Enigmatic Empirical Engines!!"
date: 2022-12-13T01:06:16+05:30
draft: true
math: true
---

## Introduction
In machine learning and statistics, feature selection, also known as variable selection, attribute selection or variable subset selection, is the process of selecting a subset of relevant features (variables, predictors) for use in model construction.

Feature Selection is a technique in which we want to decrease the number of features keeping the performance of our model same (or increasing it). It is a common perception in ML community that garbage in garbage out, so if we input noise in out model then our model is gonna return dubious outputs. So we want to create a minimal feature set from our existing feature set.

First we calculate the collinearity between a single feature and target variable (should be maximum) and we use different filter methods to calculate these collinearity:

## Notation
* $x^{(i)}$ denotes the input variables , and $y^{(i)}$ denotes the output or target variable that we’re trying to predict, where the superscript $“(i)”$ denotes the $i^{th}$ training sample.
    * Note that the superscript “(i)” in the notation is simply an index into the training set, and has nothing to do with exponentiation.
* A pair $(x^{(i)},y^{(i)})$ denotes a single training example.
* $x^{(i)} \in \mathbb{R}^n$, i.e., $x^{(i)}$ is $n$ dimensional, while $y^{(i)} \in \mathcal{C}$.
* $x^{(i)}_j$ represents the $j^{th}$ feature of the $i^{th}$ training sample.
* $m$ denotes the number of training examples.
* $n$ denotes the number of features.
* The entire training data is denoted as $\mathcal{D} = \lbrace ({x}^{(1)},y^{(1)}), \cdots, ({x}^{(m)},y^{(m)}) \rbrace \subseteq \mathbb{R}^n \times \mathcal{C}$.
The data points $(\mathbf{x}_i,y_i)$ are drawn from some (unknown) distribution $P(X,Y)$. Ultimately we would like to learn a function $h$ such that for a new pair $(\mathbf{x},y)∼P$, we have $h\mathbf{(x)}=y$ with high probability (or $h(\mathbf{x})≈y$). We will get to this later. For now let us go through some examples of $X$ and $Y$.
|   |   |   |
|---|---|---|
|Binary Classification   | $\mathcal{C} = \lbrace 0,1 \rbrace$ or $\mathcal{C} = \lbrace -1,+1 \rbrace$ | Eg. spam filtering. An email is either spam (+1), or not (−1).  |
|Multi-class Classification   | $\mathcal{C} = \lbrace 1,2, \cdots, K \rbrace (K \geq 2)$  | Eg. face classification. A person can be exactly one of $K$ identities (e.g., 1="Barack Obama", 2="George W. Bush", etc.).  |
|Regression   | $\mathcal{C} = \mathbb{R}$ | Eg. predict future temperature or the height of a person. |  

$$X = \begin{bmatrix}
x_1^{(1)} & x_2^{(1)} & \dots & x_n^{(1)} \\\\
x_1^{(2)} & x_2^{(2)} & \dots & x_n^{(2)} \\\\
\vdots & \dots & \ddots & \vdots \\\\
x_1^{(m)} & \dots & \dots & x_n^{(m)}
\end{bmatrix}, Y=
\begin{bmatrix}
y^{(1)}  \\\\
y^{(2)}  \\\\
\vdots  \\\\
y^{(m)}
\end{bmatrix}$$

## Filter Methods
In terms of computation, they are very fast and inexpensive and are very good for removing duplicated, correlated, redundant features but these methods do not remove multicollinearity. Selection of feature is evaluated individually which can sometimes help when features are in isolation (don’t have a dependency on other features) but will lag when a combination of features can lead to increase in the overall performance of the model. The list of different filter methods that can be used are classified below:

We will discuss the important methods below:

### ANalysis Of VAriance (ANOVA)
The correlation between a numerical and categorical variable is find using statistical test such as F-test (ANOVA), t-test, etc. In this case, we will use ANOVA for hypothesis testing.

ANOVA stands for Analysis Of Variance. So, basically this test measures if there are any significant differences between the means of the values of the numeric variable for each categorical value. This is something that you can visualize using a box-plot as well.

Below items must be remembered about ANOVA hypothesis test:

**Null hypothesis**:
* There is no relationship between independent variable and dependent variable (basic definition)
* The variables are not correlated with each other.
* Groups means are equal (no variation in means of groups)

**Alternate hypothesis**:

* There is a relationship between independent variable and dependent variable
* The variable are correlated with each other.
* At least, one group mean is different from other groups

We then calculate the mean for all groups: The proof will be presented later but remember that we have an important identity as:
$$SS_{total} = SS_{within-group}+SS_{between-group}$$
where $SS$ represents “Sum of Squares”. After that we create what we call an ANOVA table:
$$\begin{array}{c:c:c:ccccc}
\text{Source} & \text{(DoF)} & SS & \text{Variance} \\\\ \hline
\text{within-group} & N-K & SS_{within-group} & \frac{SS_{within-group}}{N-K} \\\\ \hline
\text{between-group} & K-1 & SS_{between-group} & \frac{SS_{between-group}}{K-1} \\\\ \hline
\text{Total} & N-1 & SS_{total} & \frac{SS_{total}}{N-1}
\end {array}$$

The term F-test or F-statistic is based on the fact that these tests use the F-values to test the hypotheses. An F-values is the ratio of two variances.

Variances measure the dispersal of the data points around the mean. Higher variances occur when the individual data points tend to fall further from the mean. Here’s the F-value for one-way ANOVA.

$$F_{value}=\frac{\text{between-group variance}}{\text{within-group variance}}=\frac{\frac{SS_{between-group}}{K-1}}{\frac{SS_{within-group}}{N-K}}$$

Now, we have calculated the F-value, what we need to do is find the critical value using the F-distribution curve, The F-distribution table is organized based on the $\alpha$ value (usually 0.05). Next, the columns of the f-distribution table are based on df1 (degree of freedom for numerator or between-group) while the rows are based on df2 (degree of freedom for denominator or within-group).

Now, we have find the F-critical and F-value, then we can find whether to accept the null hypothesis or reject the null hypothesis.

### Chi-square test for Categorical Features
The chi-square statistics is a way to check the relationship between two categorical nominal variables.

Nominal variables contains values that have no intrinsic ordering. Examples of nominal variables are sex, race, eye color, skin color, etc. Ordinal variables, on the other hand, contains values that have ordering. Examples of ordinal variables are grade, education level, economic status, etc.

Pearson's chi-squared test is used to assess three types of comparison: goodness of fit, homogeneity, and independence.

* A test of goodness of fit establishes whether an observed frequency distribution differs from a theoretical distribution.
* A test of homogeneity compares the distribution of counts for two or more groups using the same categorical variable (e.g. choice of activity—college, military, employment, travel—of graduates of a high school reported a year after graduation, sorted by graduation year, to see if number of graduates choosing a given activity has changed from class to class, or from decade to decade).
* A test of independence assesses whether observations consisting of measures on two variables, expressed in a contingency table, are independent of each other (e.g. polling responses from people of different nationalities to see if one's nationality is related to the response).

For all three tests, the computational procedure includes the following steps:

1. Define your null and alternative hypotheses before collecting your data. (In our case, the null and alternative hypotheses are same from the ANOVA hypothesis testing)

2. Calculate the chi-square test statistics, using the contigency table. Here is how you create a contigency table and calculate the chi-square value using the contigency table:

$$\chi^2=\sum_{i=1}^{N}\frac{(O_i-E_i)^2}{E_i}=\sum_{i=1}^{N}\frac{O_i^2}{E_i}-N$$

3. Determine the degrees of freedom, df, of that statistic.
    * For a test of goodness-of-fit, df = (number of categories - 1)
    * For a test of homogeneity, df = (number of rows − 1)×(number of Cols − 1), where Rows corresponds to the number of categories (i.e. rows in the associated contingency table), and Cols corresponds to the number of independent groups (i.e. columns in the associated contingency table).
    * For a test of independence, df = (number of rows − 1)×(number of Cols − 1), where in this case, Rows corresponds to the number of categories in one variable, and Cols corresponds to the number of categories in the second variable.

4. Select a desired level of confidence (significance level, p-value, or the corresponding alpha level) for the result of the test.

5. Compare the obtaines chi-square value to the critical value from the chi-squared distribution with df degrees of freedom and the selected confidence level (one-sided, since the test is only in one direction, i.e. is the test value greater than the critical value?)

6. Now, we have find the chi-square-critical and chi-square-value, then we can find whether to accept the null hypothesis or reject the null hypothesis.

The above method is one form of chi-square test known as Pearson chi-square test. There are other chi-square test as well such as Yates’s correction for continuity, phi coefficient, contigency coefficient, Cramer’s V, etc. The definition of these are:
$$\chi_{yates}^2=\sum_{i=1}^{N}\frac{(|O_i-E_i|-0.5)^2}{E_i}$$
$$\phi=\pm \sqrt{\frac{\chi^2}{N}}$$
$$C=\sqrt{\frac{\chi^2}{\chi^2+N}}$$
$$V=\sqrt{\frac{\chi^2/N}{min(k-1,l-1)}}$$
