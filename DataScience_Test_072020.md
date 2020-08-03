Data Science Test July 2020
================
Adnan Fiaz

This document outlines the answer to the Data Science Test as described
in *README.md*. In short, the task is to build a model to predict the
death of a set of patients. We will proceed by first performing basic
data analysis to clean the data and then perform a number of model
iterations including feature engineering if necessary.

``` r
library(tidyverse)
library(tidymodels)
```

## Cleaning data

We start by reading in the data and calculating summary statistics. We
already know *Death* is a categorical variable so we force it to a
factor, along with all categorical variables.

``` r
raw_data <- read_csv('simulated_data.csv') %>%
  mutate(Death = factor(Death, labels=c("survived", "died"))) %>%
  mutate_if(is.character, as.factor)

raw_data
```

    ## # A tibble: 300 x 6
    ##       ID Organisation   Age   LOS Death    Category
    ##    <dbl> <fct>        <dbl> <dbl> <fct>    <fct>   
    ##  1     1 Trust1          55     2 survived Low     
    ##  2     2 Trust2          27     1 survived Low     
    ##  3     3 Trust3          93    12 survived High    
    ##  4     4 Trust4          45     3 died     Low     
    ##  5     5 Trust5          70    11 survived High    
    ##  6     6 Trust6          60     7 survived Moderate
    ##  7     7 Trust7          25     4 survived Moderate
    ##  8     8 Trust8          48     4 survived Low     
    ##  9     9 Trust9          51     7 died     Low     
    ## 10    10 Trust10         81     1 survived High    
    ## # … with 290 more rows

``` r
library(skimr)
skim(raw_data)
```

|                                                  |           |
| :----------------------------------------------- | :-------- |
| Name                                             | raw\_data |
| Number of rows                                   | 300       |
| Number of columns                                | 6         |
| \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_   |           |
| Column type frequency:                           |           |
| factor                                           | 3         |
| numeric                                          | 3         |
| \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ |           |
| Group variables                                  | None      |

Data summary

**Variable type:
factor**

| skim\_variable | n\_missing | complete\_rate | ordered | n\_unique | top\_counts                        |
| :------------- | ---------: | -------------: | :------ | --------: | :--------------------------------- |
| Organisation   |          0 |              1 | FALSE   |        10 | Tru: 30, Tru: 30, Tru: 30, Tru: 30 |
| Death          |          0 |              1 | FALSE   |         2 | sur: 247, die: 53                  |
| Category       |          0 |              1 | FALSE   |         3 | Low: 160, Mod: 87, Hig: 53         |

**Variable type:
numeric**

| skim\_variable | n\_missing | complete\_rate |   mean |    sd | p0 |   p25 |   p50 |    p75 | p100 | hist  |
| :------------- | ---------: | -------------: | -----: | ----: | -: | ----: | ----: | -----: | ---: | :---- |
| ID             |          0 |              1 | 150.50 | 86.75 |  1 | 75.75 | 150.5 | 225.25 |  300 | ▇▇▇▇▇ |
| Age            |          0 |              1 |  50.66 | 27.88 |  5 | 24.00 |  54.0 |  75.25 |   95 | ▇▅▆▆▇ |
| LOS            |          0 |              1 |   4.94 |  3.62 |  1 |  2.00 |   4.0 |   7.00 |   18 | ▇▃▂▁▁ |

As mentioned in the instructions there are indeed 300 observations and 6
columns. Starting with the outcome column, *Death*, we see there is an
imbalance in the data with the majority of the observations belonging to
the *survived* category. We also see there is an equal number of
observations from each of the 10 values in *Organisation*. Finally, the
majority of the observations belong to the *Low* value from the
*Category* variable but the number of observations in the other values
is not insignificant.

As for the numeric variables, we see that *Age* has a very wide range
from 5 to 95 but a fairly even distribution. However, the length-of-stay
or *LOS* variable is skewed to the left with a majority of the
observations having a short length-of-stay. Without any further
information we assume the values in *LOS* are acceptable.

Overall, there are no missing or out of the ordinary values so we
proceed without any cleaning of the data.

### Data Analysis

We continue with the analysis with some basic plots to uncover any
insights that might help us in the modeling steps. First, a look at the
categorical variables in relation to the outcome
variable.

![](DataScience_Test_072020_files/figure-gfm/count%20by%20organisation-1.png)<!-- -->

Since there’s an equal number of observations in each organisation
splitting by the outcome variables makes for an easy comparison. The
proportion of survived patients clearly varies by organisation. However
we also see some organisations having roughly the same proportion such
as *Trust4*, *Trust6*, *Trust9* and *Trust10*. To reduce the levels of
this categorical variable it might be useful to cluster some
organisations although I think there isn’t enough information in the
dataset to do that. Clustering would also be useful if more
organisations were added in the future, rather than retraining the model
you could then just assign any new organisation to an appropriate
cluster.

![](DataScience_Test_072020_files/figure-gfm/count%20by%20category-1.png)<!-- -->

For the *Category* variable we look at the proportion of patients as
their quantity differs by value. Here we see that a *Low* risk indeed
means a lower proportion of deaths. The proportion isn’t very different
for the *High* and *Moderate* values. It will be interesting to see how
that impacts the modeling.

Since we mentioned clustering the organisations it is perhaps
interesting to see how *Category* differs by
organisation.

![](DataScience_Test_072020_files/figure-gfm/proportion%20by%20organisation%20and%20category-1.png)<!-- -->

There are organisations which have similar proportions of the *Category*
variable such as *Trust3* and *Trust4* but their proportion of the
outcome variable still differs. Also *Trust7* springs out with
significantly different proportions yet its proportion of the outcome
variable is only slightly different from other organisations.

We continue looking at the numeric
variables.

![](DataScience_Test_072020_files/figure-gfm/boxplot%20age-1.png)<!-- -->

For the *Age* variable we see that for the *died* outcome the
distribution is skewed towards higher values. However the distribution
for the *survived* outcome seems to be bimodal. It looks like our
earlier observation of a fairly even *Age* distribution wasn’t entirely
true, the overall distribution looks closer to a bimodal distribution
with a peak at young and old
ages.

![](DataScience_Test_072020_files/figure-gfm/boxplot%20los-1.png)<!-- -->

For the *LOS* variable we also see different distributions for the two
outcome. For the *survived* value the distribution is similar to the
overall distribution, namely left skewed. For the *died* value the
distribution is more stretched out and exhibits more values for longer
length of stays.

Altogether, this basic data analysis hasn’t uncovered anything unusual
requiring us to clean or manipulate the data. The variables all seem to
have some predictive value so we don’t need to drop any of them either.
We could do further analysis of the data looking at more cross-sections
(e.g. Age by Category) but we have sufficient information to proceed to
the modelling step so we’ll leave that for now.

## Modelling data

Granted we could’ve skipped the above data analysis and gone straight to
building a model but it’s equally important to get to know the data (and
make pretty plots in the process). Now that we are here, we first need
to establish how we will model this challenge. Since the outcome
variable is binary we will need to build a classification model. As for
the metric to assess model performance, we will use ROC and AUC as these
are “assumption-free” metrics. Metrics such as F1-score or accuracy
require assumptions about the relative importance of the outcomes and/or
probability cut-off.

To get an unbiased value for the metric we split the data into a
training and test set.

``` r
set.seed(20200803)

raw_data_split <- initial_split(raw_data, prop = 0.8, strata=Death)

training_data <- training(raw_data_split)
test_data <- testing(raw_data_split)
```

### Model Iteration 1: The baseline

Now that we know where we’re headed let’s see how we can get there
really quick. It’s important to establish a simple baseline model before
we start adding complexity. The simplest classification model that I can
think of is a logistic regression.

``` r
# define preprocessing (none in this case)
baseline_recipe <- 
  recipe(Death ~ ., data=training_data) %>%
  update_role(ID, new_role="ID") # ensures the ID isn't used in the model without removing it

# define model
baseline_model <- 
  logistic_reg() %>%
  set_engine("glm")

# bundle model and preprocessing
baseline_wf <- workflow() %>%
  add_recipe(baseline_recipe) %>%
  add_model(baseline_model)

baseline_wf
```

    ## ══ Workflow ═════════════════════════════════════════════════════════════════════════════════════════════════════════════════
    ## Preprocessor: Recipe
    ## Model: logistic_reg()
    ## 
    ## ── Preprocessor ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
    ## 0 Recipe Steps
    ## 
    ## ── Model ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    ## Logistic Regression Model Specification (classification)
    ## 
    ## Computational engine: glm

The above seems a bit convoluted but using the **tidymodels** framework
does make it easier to iterate over models as we will see later. Next we
will fit the model using 3-fold cross validation. We could just fit it
once but with cross validation we get a better estimate of the unbiased
metric.

``` r
set.seed(20200803) # set random generator seed to ensure reproducibility

baseline_wf %>%
  fit_resamples(vfold_cv(training_data, v=3, strata=Death),
                metrics = metric_set(roc_auc)) %>%
  collect_metrics() %>%
  mutate(model="baseline") %>%
  select(model, auc=mean) -> leaderboard

leaderboard
```

    ## # A tibble: 1 x 2
    ##   model      auc
    ##   <chr>    <dbl>
    ## 1 baseline 0.560

So our baseline model performs better than a model where the outcome is
randomly chosen (AUC=0.5). That’s great but let’s see if we can improve
on this.

### Model Iteration 2: Preprocessing

Preprocessing is a type of feature engineering, we’re providing
information to the model on how to better learn from the input
variables. In this case we first need to convert the categorical
variables to dummy variables. Our numerical variables, *Age* and *LOS*
are also on different scales so we can center and scale these. We will
do all of this by adjusting the *recipe*.

``` r
preproces_recipe <- baseline_recipe %>%
  step_normalize(all_numeric()) %>%
  step_dummy(all_nominal(), -all_outcomes()) %>%
  step_zv(all_predictors()) # remove any variables that have zero variance

preprocess_wf <-
  baseline_wf %>%
  update_recipe(preproces_recipe)
```

``` r
set.seed(20200803) # set seed again to ensure we get the same CV folds
# at this point you should make a function of this
preprocess_wf %>%
  fit_resamples(vfold_cv(training_data, v=3, strata=Death),
                metrics = metric_set(roc_auc)) %>%
  collect_metrics() %>%
  mutate(model="preprocess_lr") %>%
  select(model, auc=mean) %>%
  bind_rows(leaderboard) -> leaderboard

leaderboard
```

    ## # A tibble: 2 x 2
    ##   model           auc
    ##   <chr>         <dbl>
    ## 1 preprocess_lr 0.560
    ## 2 baseline      0.560

It seems that preprocessing doesn’t have any added benefit. This could
be because of two reasons: there is no actual difference in the model
(i.e. there is preprocessing happening in the baseline model) or there
is no actual difference in performance. The example
[documentation](https://www.tidymodels.org/start/recipes/#features)
states the baseline recipe we created doesn’t do any preprocessing so
we’ll go with the second reason. We could investigate the model fits
(using `pull_worklfow_fit`) to be sure but I’ll leave that for another
time as this isn’t the final model iteration.

### Model Iteration 3: The Random Forest

Now is the time to add some more complexity. We could go two ways here,
add more features or try another algorithm. Trying another algorithm is
quicker to do so we’ll start with that. Since we’ll be using a
tree-based algorithm we’ll skip the preprocessing as tree-based
algorithms are flexible enough to handle the data without it.

``` r
rf_model <- 
  rand_forest() %>%
  set_engine(engine='ranger') %>%
  set_mode('classification')

rf_wf <- 
  baseline_wf %>%
  update_model(rf_model)
```

``` r
set.seed(20200803)
# I really wish I had made a function of this
rf_wf %>%
  fit_resamples(vfold_cv(training_data, v=3, strata=Death),
                metrics = metric_set(roc_auc)) %>%
  collect_metrics() %>%
  mutate(model="rf") %>%
  select(model, auc=mean) %>%
  bind_rows(leaderboard) -> leaderboard

leaderboard
```

    ## # A tibble: 3 x 2
    ##   model           auc
    ##   <chr>         <dbl>
    ## 1 rf            0.641
    ## 2 preprocess_lr 0.560
    ## 3 baseline      0.560

The Random Forest outperforms the logistic regression based on the AUC.
Now we can train it on the entire dataset and apply it to the test data
to get a final performance value.

``` r
set.seed(20200803)
rf_wf %>%
  last_fit(split=raw_data_split, metrics=metric_set(roc_auc)) %>%
  collect_metrics()
```

    ## # A tibble: 1 x 3
    ##   .metric .estimator .estimate
    ##   <chr>   <chr>          <dbl>
    ## 1 roc_auc binary         0.718

## Follow up

We performed three model iterations in which the Random Forest performed
best. There are various reasons why it performed better than the
logistic regression model but clearly there is some complexity or
information in the data that isn’t being captured by the logistic
regression model. Possible next steps are:

  - Analyse performance of the best model by looking at which
    observations are missclassified. What information is not being
    captured? Which variables does the model find are most important?
    Since Random Forests are black box models we could supplement this
    analysis with LIME and/or Shapley values to find out how predictions
    were made.
  - Add more features. Although the Random Forest should be able to
    capture any non-linear information we could make things more
    explicit. For example, we know Age has a bimodal distribution and we
    can use this to create new features.
  - Since the dataset was imbalanced with respect to the outcome
    variable we could try balancing the dataset by upsampling or
    downsampling.
  - Tune hyperparameters for the best model. In the case of the Random
    Forest that could be the number of trees (`trees`) or the number of
    variables to be considered at each split (`mtry`).

## Addendum: Model Cards

I recently came across the concept of [Model
Cards](https://modelcards.withgoogle.com/about) and I’m trying to apply
it in practice wherever possible. What particularly attracts me is the
importance of describing the limits of the model and the model data.
Below is an initial attempt at creating a model card for the Random
Forest.

<!--html_preserve-->

<style>html {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Helvetica Neue', 'Fira Sans', 'Droid Sans', Arial, sans-serif;
}

#grqinhhnoc .gt_table {
  display: table;
  border-collapse: collapse;
  margin-left: 0;
  margin-right: auto;
  color: #333333;
  font-size: 16px;
  background-color: #FFFFFF;
  width: auto;
  border-top-style: solid;
  border-top-width: 2px;
  border-top-color: black;
  border-right-style: solid;
  border-right-width: 2px;
  border-right-color: black;
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: black;
  border-left-style: solid;
  border-left-width: 2px;
  border-left-color: black;
}

#grqinhhnoc .gt_heading {
  background-color: #FFFFFF;
  text-align: center;
  border-bottom-color: #FFFFFF;
  border-left-style: none;
  border-left-width: 1px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 1px;
  border-right-color: #D3D3D3;
}

#grqinhhnoc .gt_title {
  color: #333333;
  font-size: 125%;
  font-weight: initial;
  padding-top: 4px;
  padding-bottom: 4px;
  border-bottom-color: #FFFFFF;
  border-bottom-width: 0;
}

#grqinhhnoc .gt_subtitle {
  color: #333333;
  font-size: 85%;
  font-weight: initial;
  padding-top: 0;
  padding-bottom: 4px;
  border-top-color: #FFFFFF;
  border-top-width: 0;
}

#grqinhhnoc .gt_bottom_border {
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: black;
}

#grqinhhnoc .gt_col_headings {
  border-top-style: solid;
  border-top-width: 2px;
  border-top-color: #D3D3D3;
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
  border-left-style: none;
  border-left-width: 1px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 1px;
  border-right-color: #D3D3D3;
}

#grqinhhnoc .gt_col_heading {
  color: #333333;
  background-color: #FFFFFF;
  font-size: 100%;
  font-weight: normal;
  text-transform: inherit;
  border-left-style: none;
  border-left-width: 1px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 1px;
  border-right-color: #D3D3D3;
  vertical-align: bottom;
  padding-top: 5px;
  padding-bottom: 6px;
  padding-left: 5px;
  padding-right: 5px;
  overflow-x: hidden;
}

#grqinhhnoc .gt_column_spanner_outer {
  color: #333333;
  background-color: #FFFFFF;
  font-size: 100%;
  font-weight: normal;
  text-transform: inherit;
  padding-top: 0;
  padding-bottom: 0;
  padding-left: 4px;
  padding-right: 4px;
}

#grqinhhnoc .gt_column_spanner_outer:first-child {
  padding-left: 0;
}

#grqinhhnoc .gt_column_spanner_outer:last-child {
  padding-right: 0;
}

#grqinhhnoc .gt_column_spanner {
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
  vertical-align: bottom;
  padding-top: 5px;
  padding-bottom: 6px;
  overflow-x: hidden;
  display: inline-block;
  width: 100%;
}

#grqinhhnoc .gt_group_heading {
  padding: 8px;
  color: #333333;
  background-color: #FFFFFF;
  font-size: 100%;
  font-weight: initial;
  text-transform: inherit;
  border-top-style: solid;
  border-top-width: 2px;
  border-top-color: #D3D3D3;
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
  border-left-style: none;
  border-left-width: 1px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 1px;
  border-right-color: #D3D3D3;
  vertical-align: middle;
}

#grqinhhnoc .gt_empty_group_heading {
  padding: 0.5px;
  color: #333333;
  background-color: #FFFFFF;
  font-size: 100%;
  font-weight: initial;
  border-top-style: solid;
  border-top-width: 2px;
  border-top-color: #D3D3D3;
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
  vertical-align: middle;
}

#grqinhhnoc .gt_striped {
  background-color: rgba(128, 128, 128, 0.05);
}

#grqinhhnoc .gt_from_md > :first-child {
  margin-top: 0;
}

#grqinhhnoc .gt_from_md > :last-child {
  margin-bottom: 0;
}

#grqinhhnoc .gt_row {
  padding-top: 8px;
  padding-bottom: 8px;
  padding-left: 5px;
  padding-right: 5px;
  margin: 10px;
  border-top-style: none;
  border-top-width: 1px;
  border-top-color: #D3D3D3;
  border-left-style: none;
  border-left-width: 1px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 1px;
  border-right-color: #D3D3D3;
  vertical-align: middle;
  overflow-x: hidden;
}

#grqinhhnoc .gt_stub {
  color: #333333;
  background-color: #FFFFFF;
  font-size: 100%;
  font-weight: initial;
  text-transform: inherit;
  border-right-style: solid;
  border-right-width: 2px;
  border-right-color: #D3D3D3;
  padding-left: 12px;
}

#grqinhhnoc .gt_summary_row {
  color: #333333;
  background-color: #FFFFFF;
  text-transform: inherit;
  padding-top: 8px;
  padding-bottom: 8px;
  padding-left: 5px;
  padding-right: 5px;
}

#grqinhhnoc .gt_first_summary_row {
  padding-top: 8px;
  padding-bottom: 8px;
  padding-left: 5px;
  padding-right: 5px;
  border-top-style: solid;
  border-top-width: 2px;
  border-top-color: #D3D3D3;
}

#grqinhhnoc .gt_grand_summary_row {
  color: #333333;
  background-color: #FFFFFF;
  text-transform: inherit;
  padding-top: 8px;
  padding-bottom: 8px;
  padding-left: 5px;
  padding-right: 5px;
}

#grqinhhnoc .gt_first_grand_summary_row {
  padding-top: 8px;
  padding-bottom: 8px;
  padding-left: 5px;
  padding-right: 5px;
  border-top-style: double;
  border-top-width: 6px;
  border-top-color: #D3D3D3;
}

#grqinhhnoc .gt_table_body {
  border-top-style: solid;
  border-top-width: 2px;
  border-top-color: #D3D3D3;
  border-bottom-style: none;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
}

#grqinhhnoc .gt_footnotes {
  color: #333333;
  background-color: #FFFFFF;
  border-bottom-style: none;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
  border-left-style: none;
  border-left-width: 2px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 2px;
  border-right-color: #D3D3D3;
}

#grqinhhnoc .gt_footnote {
  margin: 0px;
  font-size: 90%;
  padding: 4px;
}

#grqinhhnoc .gt_sourcenotes {
  color: #333333;
  background-color: #FFFFFF;
  border-bottom-style: none;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
  border-left-style: none;
  border-left-width: 2px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 2px;
  border-right-color: #D3D3D3;
}

#grqinhhnoc .gt_sourcenote {
  font-size: 90%;
  padding: 4px;
}

#grqinhhnoc .gt_left {
  text-align: left;
}

#grqinhhnoc .gt_center {
  text-align: center;
}

#grqinhhnoc .gt_right {
  text-align: right;
  font-variant-numeric: tabular-nums;
}

#grqinhhnoc .gt_font_normal {
  font-weight: normal;
}

#grqinhhnoc .gt_font_bold {
  font-weight: bold;
}

#grqinhhnoc .gt_font_italic {
  font-style: italic;
}

#grqinhhnoc .gt_super {
  font-size: 65%;
}

#grqinhhnoc .gt_footnote_marks {
  font-style: italic;
  font-size: 65%;
}
</style>

<div id="grqinhhnoc" style="overflow-x:auto;overflow-y:auto;width:auto;height:auto;">

<table class="gt_table">

<thead class="gt_header">

<tr>

<th colspan="2" class="gt_heading gt_title gt_font_normal" style>

Model Card Predict Patient
Death

</th>

</tr>

<tr>

<th colspan="2" class="gt_heading gt_subtitle gt_font_normal gt_bottom_border" style>

</th>

</tr>

</thead>

<tbody class="gt_table_body">

<tr>

<td class="gt_row gt_left" style="font-weight: bold;">

Goal:

</td>

<td class="gt_row gt_left">

Predict whether a patient will survive

</td>

</tr>

<tr>

<td class="gt_row gt_left" style="font-weight: bold;">

Input:

</td>

<td class="gt_row gt_left">

Organisation, Category, Age and Length of Stay (LOS)

</td>

</tr>

<tr>

<td class="gt_row gt_left" style="font-weight: bold;">

Output

</td>

<td class="gt_row gt_left">

A value from {“survived”, “died”}

</td>

</tr>

<tr>

<td class="gt_row gt_left" style="font-weight: bold;">

Model Architecture:

</td>

<td class="gt_row gt_left">

Random Forest trained with the {ranger} package, default settings

</td>

</tr>

<tr>

<td class="gt_row gt_left" style="font-weight: bold;">

Performance:

</td>

<td class="gt_row gt_left">

0.71 Test AUC (59 observations)

</td>

</tr>

<tr>

<td class="gt_row gt_left" style="font-weight: bold;">

Limitations:

</td>

<td class="gt_row gt_left">

5 \<= Age \<= 95

</td>

</tr>

<tr>

<td class="gt_row gt_left" style="font-weight: bold;">

</td>

<td class="gt_row gt_left">

1 \<= LOS \<= 18

</td>

</tr>

<tr>

<td class="gt_row gt_left" style="font-weight: bold;">

</td>

<td class="gt_row gt_left">

10 Organisations

</td>

</tr>

<tr>

<td class="gt_row gt_left" style="font-weight: bold;">

</td>

<td class="gt_row gt_left">

3 Categories

</td>

</tr>

</tbody>

</table>

</div>

<!--/html_preserve-->
