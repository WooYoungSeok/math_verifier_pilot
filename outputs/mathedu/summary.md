# MathEdu gold-label evaluation

gold rows: 539 (balanced subset: 202, seed 42)

## Overview

```
       model                   subset   n  accuracy  balanced_accuracy  macro_f1  coverage  accuracy_decided  recall_concept_gap  recall_slip
gpt-5.4-mini                 balanced 202     0.822              0.822     0.820     0.995             0.826               0.980        0.663
gpt-5.4-mini                     full 539     0.918              0.820     0.863     0.991             0.927               0.977        0.663
gpt-5.4-mini full, problem_match only 403     0.923              0.832     0.872     0.995             0.928               0.984        0.679
     gpt-5.1                 balanced 202     0.812              0.812     0.816     0.980             0.828               0.960        0.663
     gpt-5.1                     full 539     0.909              0.815     0.857     0.981             0.926               0.966        0.663
     gpt-5.1 full, problem_match only 403     0.911              0.815     0.855     0.990             0.920               0.975        0.654
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
