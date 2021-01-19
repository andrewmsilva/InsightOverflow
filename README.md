# An exploratory analysis employing topic modeling: Tracking evolution and loyalty from Stack Overflow users' interests


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
  Corpus built (04:00:36.19)
  Experiment done: lda, 20, 5000, 10 | 11:23:51.30, 0.4516
  Experiment done: nmf, 20, 5000, 10 | 09:36:08.56, 0.4633
  Experiment done: lda, 40, 5000, 10 | 11:13:32.81, 0.4541
  Experiment done: nmf, 40, 5000, 10 | 10:45:23.85, 0.4426
  Experiment done: lda, 60, 5000, 10 | 12:01:09.44, 0.4568
  Experiment done: nmf, 60, 5000, 10 | 17:37:11.10, 0.3690
  Experiment done: lda, 80, 5000, 10 | 11:08:15.73, 0.4243
  Experiment done: nmf, 80, 5000, 10 | 38:21:01.85, 0.3591
  Experiment done: lda, 20, 50000, 10 | 12:19:49.57, 0.4598
  Experiment done: nmf, 20, 50000, 10 | 10:06:03.60, 0.4597
  Experiment done: lda, 40, 50000, 10 | 12:07:26.47, 0.4550
  Experiment done: nmf, 40, 50000, 10 | 10:56:47.65, 0.4440
  Experiment done: lda, 60, 50000, 10 | 11:38:48.09, 0.4503
  Experiment done: nmf, 60, 50000, 10 | 12:30:02.23, 0.3775
  Experiment done: lda, 80, 50000, 10 | 11:35:21.22, 0.4425
  Experiment done: nmf, 80, 50000, 10 | 40:52:54.07, 0.3673
  Experiment done: lda, 20, 500000, 10 | 11:41:04.00, 0.4564
  Experiment done: nmf, 20, 500000, 10 | 10:07:15.20, 0.4430
  Experiment done: lda, 40, 500000, 10 | 11:43:42.61, 0.4500
  Experiment done: nmf, 40, 500000, 10 | 11:29:37.91, 0.4430
  Experiment done: lda, 60, 500000, 10 | 11:39:44.58, 0.4632
  Experiment done: nmf, 60, 500000, 10 | 14:33:41.75, 0.3747
  Experiment done: lda, 80, 500000, 10 | 12:56:36.62, 0.4631
  Experiment done: nmf, 80, 500000, 10 | 53:32:39.98, 0.3752
  Experiment done: lda, 20, 5000, 100 | 113:15:56.04, 0.4518
```