# An exploratory analysis employing topic modeling: Tracking evolution and loyalty from Stack Overflow users' interests


Running this experiment requires downloading Stack Overflow posts from the [data dump](https://archive.org/download/stackexchange/stackoverflow.com-Posts.7z) and extract the `.7z` file into ```src/data/```. As this algorithm employs Redis database for extraction step, installing, configuring, and starting Redis is essential (a tutorial is found [here](https://redis.io/topics/quickstart)).

## Extraction

```
Extraction started
  Extracted: 49598818
  Ignored: 739023
  Total: 50337841
Execution time: 05:39:07.74
```

## Pre-processing

```
Pre-processing started
Execution time: 74:53:18.44
```