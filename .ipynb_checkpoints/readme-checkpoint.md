# GIOU python


## Installation

```
	cd giou_cpp
	python setup.py install (--user)
```

## Example

```
Run:
    python test.py


The result is:
     iou cpu using: 0.00s
    giou cpu using: 0.01s
    py iou using: 0.37s
    py giou using: 0.64s
    
```


```
Run:
    python test2.py


The result is:
    iou =  tensor([[0.5769]])
    giou =  tensor([[0.5399]])
    py iou =  0.5769230980807928
    py giou =  0.5398860418083425
```

