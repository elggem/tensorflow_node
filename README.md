# ae-destin

This is a tensorflow based implementation of the DeSTIN perceptional framework using Stacked Denoising Autoencoders in each of the nodes. It aims to be a flexible implemention that can be modified and inspected during runtime on live stream data. As of now the code is used for research purposes, so handle with care!

As of this moment the code is not optimized for GPU usage. Potentially this can be heavily optimized in the future.

## I want to run it!

Clone the repo, install tensorflow on your machine and execute the following from the root directory:

```
python tests/callbacks.py
```

TF summaries are being written to `outputs/summaries`, and they can be inspected via this command:

```
tensorboard --logdir=runs:output/summaries --port 6006 --reload_interval 5
```

## Participate

I've put all the todos and remaining tasks in the projects tab on Github. Feel free to collaborate or contact me if you have any suggestions!