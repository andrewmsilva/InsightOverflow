# Insight Overflow

An exploratory analysis employing topic modeling: Tracking evolution and loyalty from Stack Overflow users' interests

Running this experiment requires downloading Stack Overflow posts from the [data dump](https://archive.org/download/stackexchange/stackoverflow.com-Posts.7z) and extract the `.7z` file into ```src/data/```. As this algorithm employs Redis database for extraction step, installing, configuring, and starting Redis is essential (a tutorial is found [here](https://redis.io/topics/quickstart)).

## Extraction

```
Extraction started
  Extracted: 49598818
  Ignored: 739023
  Total: 50337841
Execution time: 04:46:22.44
```

## Pre-processing

```
Pre-processing started
Execution time: 102:39:36.14
```

## Topic modeling

```
Topic modeling started
  Corpus built: 00:00:01.65
  Experiment done: k=20 i=10 | p=4133.9019, cv=0.4946
  Experiment done: k=20 i=100 | p=1433.5471, cv=0.6330
  Experiment done: k=20 i=200 | p=1388.5460, cv=0.6343
  Experiment done: k=20 i=500 | p=1365.3670, cv=0.6341
  Experiment done: k=40 i=10 | p=5503.5514, cv=0.5449
  Experiment done: k=40 i=100 | p=1448.7289, cv=0.6046
  Experiment done: k=40 i=200 | p=1379.5958, cv=0.6051
  Experiment done: k=40 i=500 | p=1330.4556, cv=0.6072
  Experiment done: k=60 i=10 | p=6675.3963, cv=0.5221
  Experiment done: k=60 i=100 | p=1448.0626, cv=0.5874
  Experiment done: k=60 i=200 | p=1349.6507, cv=0.5940
  Experiment done: k=60 i=500 | p=1290.6926, cv=0.5880
  Experiment done: k=80 i=10 | p=7576.2664, cv=0.5115
  Experiment done: k=80 i=100 | p=1457.7716, cv=0.5800
  Experiment done: k=80 i=200 | p=1351.4062, cv=0.5866
  Experiment done: k=80 i=500 | p=1288.1277, cv=0.5892
  Experiment done: k=100 i=10 | p=8093.3122, cv=0.5114
  Experiment done: k=100 i=100 | p=1448.3062, cv=0.5762
  Experiment done: k=100 i=200 | p=1341.3547, cv=0.5787
  Experiment done: k=100 i=500 | p=1272.4512, cv=0.5794
Execution time: 00:54:22.32
```
