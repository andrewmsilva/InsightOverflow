# TCC (an appropriate title will be created soon)

## Data downloading

To run this experiment, it is necessary to download the [data](https://archive.org/download/stackexchange/stackoverflow.com-Posts.7z) and extract the .7z file into the "data" folder. 

## Data extraction
```shell
$ cd app
$ python3 extraction.py
Extraction started
  Extracted: 25980297
  Ignored: 21950804
  Total: 47931101
Done in 7351.8453 seconds
```

## Preprocessing
```shell
$ cd app
$ python3 preprocessing.py
Preprocessing started
Done in 204276.4271 seconds
```