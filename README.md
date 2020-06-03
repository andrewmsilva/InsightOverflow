# TCC (an appropriate title will be created soon)

## Downloading

To run this experiment, it is necessary to download the [data](https://archive.org/download/stackexchange/stackoverflow.com-Posts.7z) and extract the .7z file into the "data" folder. 

## Extraction

This step uses the Redis database and, therefore, it is necessary to install, configure, and start Redis before run extraction (a tutorial is found [here](https://redis.io/topics/quickstart)).

```
$ cd app
$ python3 extraction.py
Extraction started
  Extracted: 25980297
  Ignored: 21950804
  Total: 47931101
Done in 7818.8945 seconds
```

## Preprocessing
```
$ cd app
$ python3 preprocessing.py
Preprocessing started
Done in 204276.4271 seconds
```