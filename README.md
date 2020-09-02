# TCC (an appropriate title will be created soon)

## Downloading

To run this experiment, it is necessary to download the [data](https://archive.org/download/stackexchange/stackoverflow.com-Posts.7z) and extract the .7z file into the "data" folder. 

## Extraction

This step uses the Redis database and, therefore, it is necessary to install, configure, and start Redis before run extraction (a tutorial is found [here](https://redis.io/topics/quickstart)).

```
$ cd app
$ python3 extraction.py
Extraction started
  Extracted: 49036277
  Ignored: 103630
  Total: 49139907
  Elapsed time: 04:00:59.65
```

## Pre-processing

### Cleaning

```
$ cd app
$ python3 cleaning.py
Cleaning started
  Elapsed time: 66:27:11.38
```

### Enrichment

```
$ cd app
$ python3 enrichment.py
Enrichment started
  Elapsed time: 04:22:37.79
```