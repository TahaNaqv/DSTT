# Assignment #01 - Report

Data Science Tools & Techniques (MS DS)
Q#1: Data Pre-processing and EDA

Name: Taha Abbas Naqvi
Roll No: 261-8004
Class: MS (DS)
Section: A

## Task 1: Pre-processing

The two datasets are broken in different ways, so they needed different treatment.

The attached dataset has bad values. The word "unknown" was used instead of a blank in
annual_income and credit_score, so pandas read both of them as text and they did not even show up
in describe(). There were also 1509 empty cells across six columns, 49 duplicated
rows, and the same country written three ways (us, USA, U.S.). Some values are not possible at all:
hours_per_week goes up to 391 when a week only has 168 hours, BMI reaches 204, twelve salaries are
negative, and 1926 people have more years of work experience than their age minus 16.

The cleaning order was: replace the sentinel and fix the data types, drop duplicates, set out of
range values to NaN, cap the money columns at the 1st and 99th percentile, and only then impute,
using the median for numbers and the mode for categories. Imputing first would have been a mistake,
because the medians would then be calculated from the corrupt values.

The census dataset is the opposite case. Its values are fine but the structure is difficult. There
is no header row, so the column names came from census-income.names. Missing values are written as
"?" (415717 of them), and thirteen columns are full of "Not in universe", which is not missing data
at all. It means the question did not apply to that person, so I kept it as a normal category.
There are 46627 duplicate records once the survey weight is ignored, and that weight,
instance_weight, must not be used as a feature because it says how many people each row stands for.
Five more columns look numeric but are really codes, and four money columns are 89 to 98 percent
zeros.

For encoding I picked the method per column rather than using one method everywhere. Education,
smoking status and physical activity have a real order, so they got integer codes. Two level
variables got a single 0/1 column. Short nominal lists got one hot encoding with drop_first, so the
dummies are not perfectly collinear. For the nine census columns with many levels, such as the 51
states and the 43 birth countries, one hot would have added around 220 almost empty columns, so I
used frequency encoding there.

## Task 2: EDA before and after

The first EDA pass is what made the issue lists possible. info() exposed the two columns whose type
gave away the hidden sentinel, the missing value report showed where the gaps were, the boxplots
showed the impossible BMI and hour values, and the skew numbers (16.1 for annual_income, 27.8 for
dividends) showed how far the tails were pulling the averages.

The second pass shows the result: no missing cells, no duplicates, no text columns left, features
on a comparable scale, and skew down to 0.1 and 2.6. The shape of the sensible part of each
distribution is unchanged, so the cleaning removed noise and not the pattern.

One choice was made by testing instead of by habit. A log transform of income is the usual advice,
but after capping, the skew was already near zero and the log pushed it to -1.3, so I left it out.

## Why this matters

Almost a quarter of the census rows were duplicates, and left alone they would make whatever
pattern they carry look more certain than it is. One stray string turned a numeric column into
text. Features that differ by four orders of magnitude dominate any model based on distances or
gradients. Pre-processing decides whether the analysis that follows measures the data or the
mistakes in the file.
