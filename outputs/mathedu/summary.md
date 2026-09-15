# MathEdu gold-label evaluation

gold rows: 539 (balanced subset: 202, seed 42)

## Overview

```
              model                       subset   n  accuracy  balanced_accuracy  macro_f1  coverage  accuracy_decided  recall_concept_gap  recall_slip
            gpt-5.1                     balanced 202     0.812              0.812     0.816     0.980             0.828               0.960        0.663
            gpt-5.1 balanced, problem_match only 150     0.800              0.813     0.804     0.987             0.811               0.971        0.654
            gpt-5.1                         full 539     0.909              0.815     0.857     0.981             0.926               0.966        0.663
            gpt-5.1     full, problem_match only 403     0.911              0.815     0.855     0.990             0.920               0.975        0.654
       gpt-5.4-mini                     balanced 202     0.822              0.822     0.820     0.995             0.826               0.980        0.663
       gpt-5.4-mini balanced, problem_match only 150     0.820              0.832     0.819     1.000             0.820               0.986        0.679
       gpt-5.4-mini                         full 539     0.918              0.820     0.863     0.991             0.927               0.977        0.663
       gpt-5.4-mini     full, problem_match only 403     0.923              0.832     0.872     0.995             0.928               0.984        0.679
qwen2.5-7b-instruct                     balanced 202     0.683              0.683     0.677     0.990             0.690               0.861        0.505
qwen2.5-7b-instruct balanced, problem_match only 150     0.667              0.682     0.663     0.993             0.671               0.870        0.494
qwen2.5-7b-instruct                         full 539     0.783              0.676     0.675     0.983             0.796               0.847        0.505
qwen2.5-7b-instruct     full, problem_match only 403     0.769              0.666     0.665     0.988             0.779               0.839        0.494
```

## gpt-5.1

labelled rows: 539 / 539 (missing 0)

### balanced (n=202)

| accuracy | balanced acc | macro-F1 | coverage | acc (decided) | recall concept_gap | recall slip |
|---|---|---|---|---|---|---|
| 0.812 | 0.812 | 0.816 | 0.980 | 0.828 | 0.960 | 0.663 |

confusion (rows = gold, cols = predicted):

```
pred         ambiguous  concept_gap  slip
gold                                     
concept_gap          2           97     2
slip                 2           32    67
```

recall by teacher error type:

```
                                        n    recall  ambiguous
error_type                                                    
Arithmetical error                     73  0.794521   0.027397
Careless error                         28  0.321429   0.000000
Wrong mathematical operation/concept  101  0.960396   0.019802
```

### balanced, problem_match only (n=150)

| accuracy | balanced acc | macro-F1 | coverage | acc (decided) | recall concept_gap | recall slip |
|---|---|---|---|---|---|---|
| 0.800 | 0.813 | 0.804 | 0.987 | 0.811 | 0.971 | 0.654 |

confusion (rows = gold, cols = predicted):

```
pred         ambiguous  concept_gap  slip
gold                                     
concept_gap          1           67     1
slip                 1           27    53
```

recall by teacher error type:

```
                                       n    recall  ambiguous
error_type                                                   
Arithmetical error                    60  0.783333   0.016667
Careless error                        21  0.285714   0.000000
Wrong mathematical operation/concept  69  0.971014   0.014493
```

### full (n=539)

| accuracy | balanced acc | macro-F1 | coverage | acc (decided) | recall concept_gap | recall slip |
|---|---|---|---|---|---|---|
| 0.909 | 0.815 | 0.857 | 0.981 | 0.926 | 0.966 | 0.663 |

confusion (rows = gold, cols = predicted):

```
pred         ambiguous  concept_gap  slip
gold                                     
concept_gap          8          423     7
slip                 2           32    67
```

recall by teacher error type:

```
                                        n    recall  ambiguous
error_type                                                    
Arithmetical error                     73  0.794521   0.027397
Careless error                         28  0.321429   0.000000
Wrong mathematical operation/concept  438  0.965753   0.018265
```

### full, problem_match only (n=403)

| accuracy | balanced acc | macro-F1 | coverage | acc (decided) | recall concept_gap | recall slip |
|---|---|---|---|---|---|---|
| 0.911 | 0.815 | 0.855 | 0.990 | 0.920 | 0.975 | 0.654 |

confusion (rows = gold, cols = predicted):

```
pred         ambiguous  concept_gap  slip
gold                                     
concept_gap          3          314     5
slip                 1           27    53
```

recall by teacher error type:

```
                                        n    recall  ambiguous
error_type                                                    
Arithmetical error                     60  0.783333   0.016667
Careless error                         21  0.285714   0.000000
Wrong mathematical operation/concept  322  0.975155   0.009317
```

## gpt-5.4-mini

labelled rows: 539 / 539 (missing 0)

### balanced (n=202)

| accuracy | balanced acc | macro-F1 | coverage | acc (decided) | recall concept_gap | recall slip |
|---|---|---|---|---|---|---|
| 0.822 | 0.822 | 0.820 | 0.995 | 0.826 | 0.980 | 0.663 |

confusion (rows = gold, cols = predicted):

```
pred         ambiguous  concept_gap  slip
gold                                     
concept_gap          1           99     1
slip                 0           34    67
```

recall by teacher error type:

```
                                        n    recall  ambiguous
error_type                                                    
Arithmetical error                     73  0.780822   0.000000
Careless error                         28  0.357143   0.000000
Wrong mathematical operation/concept  101  0.980198   0.009901
```

### balanced, problem_match only (n=150)

| accuracy | balanced acc | macro-F1 | coverage | acc (decided) | recall concept_gap | recall slip |
|---|---|---|---|---|---|---|
| 0.820 | 0.832 | 0.819 | 1.000 | 0.820 | 0.986 | 0.679 |

confusion (rows = gold, cols = predicted):

```
pred         concept_gap  slip
gold                          
concept_gap           68     1
slip                  26    55
```

recall by teacher error type:

```
                                       n    recall  ambiguous
error_type                                                   
Arithmetical error                    60  0.766667        0.0
Careless error                        21  0.428571        0.0
Wrong mathematical operation/concept  69  0.985507        0.0
```

### full (n=539)

| accuracy | balanced acc | macro-F1 | coverage | acc (decided) | recall concept_gap | recall slip |
|---|---|---|---|---|---|---|
| 0.918 | 0.820 | 0.863 | 0.991 | 0.927 | 0.977 | 0.663 |

confusion (rows = gold, cols = predicted):

```
pred         ambiguous  concept_gap  slip
gold                                     
concept_gap          5          428     5
slip                 0           34    67
```

recall by teacher error type:

```
                                        n    recall  ambiguous
error_type                                                    
Arithmetical error                     73  0.780822   0.000000
Careless error                         28  0.357143   0.000000
Wrong mathematical operation/concept  438  0.977169   0.011416
```

### full, problem_match only (n=403)

| accuracy | balanced acc | macro-F1 | coverage | acc (decided) | recall concept_gap | recall slip |
|---|---|---|---|---|---|---|
| 0.923 | 0.832 | 0.872 | 0.995 | 0.928 | 0.984 | 0.679 |

confusion (rows = gold, cols = predicted):

```
pred         ambiguous  concept_gap  slip
gold                                     
concept_gap          2          317     3
slip                 0           26    55
```

recall by teacher error type:

```
                                        n    recall  ambiguous
error_type                                                    
Arithmetical error                     60  0.766667   0.000000
Careless error                         21  0.428571   0.000000
Wrong mathematical operation/concept  322  0.984472   0.006211
```

## qwen2.5-7b-instruct

labelled rows: 539 / 539 (missing 0)

### balanced (n=202)

| accuracy | balanced acc | macro-F1 | coverage | acc (decided) | recall concept_gap | recall slip |
|---|---|---|---|---|---|---|
| 0.683 | 0.683 | 0.677 | 0.990 | 0.690 | 0.861 | 0.505 |

confusion (rows = gold, cols = predicted):

```
pred         ambiguous  concept_gap  slip
gold                                     
concept_gap          2           87    12
slip                 0           50    51
```

recall by teacher error type:

```
                                        n    recall  ambiguous
error_type                                                    
Arithmetical error                     73  0.520548   0.000000
Careless error                         28  0.464286   0.000000
Wrong mathematical operation/concept  101  0.861386   0.019802
```

### balanced, problem_match only (n=150)

| accuracy | balanced acc | macro-F1 | coverage | acc (decided) | recall concept_gap | recall slip |
|---|---|---|---|---|---|---|
| 0.667 | 0.682 | 0.663 | 0.993 | 0.671 | 0.870 | 0.494 |

confusion (rows = gold, cols = predicted):

```
pred         ambiguous  concept_gap  slip
gold                                     
concept_gap          1           60     8
slip                 0           41    40
```

recall by teacher error type:

```
                                       n    recall  ambiguous
error_type                                                   
Arithmetical error                    60  0.516667   0.000000
Careless error                        21  0.428571   0.000000
Wrong mathematical operation/concept  69  0.869565   0.014493
```

### full (n=539)

| accuracy | balanced acc | macro-F1 | coverage | acc (decided) | recall concept_gap | recall slip |
|---|---|---|---|---|---|---|
| 0.783 | 0.676 | 0.675 | 0.983 | 0.796 | 0.847 | 0.505 |

confusion (rows = gold, cols = predicted):

```
pred         ambiguous  concept_gap  slip
gold                                     
concept_gap          9          371    58
slip                 0           50    51
```

recall by teacher error type:

```
                                        n    recall  ambiguous
error_type                                                    
Arithmetical error                     73  0.520548   0.000000
Careless error                         28  0.464286   0.000000
Wrong mathematical operation/concept  438  0.847032   0.020548
```

### full, problem_match only (n=403)

| accuracy | balanced acc | macro-F1 | coverage | acc (decided) | recall concept_gap | recall slip |
|---|---|---|---|---|---|---|
| 0.769 | 0.666 | 0.665 | 0.988 | 0.779 | 0.839 | 0.494 |

confusion (rows = gold, cols = predicted):

```
pred         ambiguous  concept_gap  slip
gold                                     
concept_gap          5          270    47
slip                 0           41    40
```

recall by teacher error type:

```
                                        n    recall  ambiguous
error_type                                                    
Arithmetical error                     60  0.516667   0.000000
Careless error                         21  0.428571   0.000000
Wrong mathematical operation/concept  322  0.838509   0.015528
```
