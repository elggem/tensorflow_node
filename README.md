# ae-destin

This is a tensorflow based implementation of the DeSTIN perceptional framework using (stacked-)denoising autoencoders in the nodes. It aims to be a flexible implemention that can be *modified and inspected during runtime on live stream data*. Eventually it will be used in conjunction with the [OpenCog](https://github.com/opencog/opencog) framework for integrated Artificial General Intelligence.

  - *This code is under heavy development and used for research purposes, so handle with care!*
  - *It is also not greatly optimized for GPU usage, yet! This can be heavily improved in the future.*

## Documentation

You can find documentation on the [wiki](https://github.com/elggem/ae-destin/wiki) tab. There are references for the network architecture and some high-level descriptions on how it works.

## Participate

I've put todos and remaining tasks in the projects tab on Github. Feel free to collaborate or contact me if you have any suggestions!

## I want to run it!

Clone the repo, [install tensorflow](https://www.tensorflow.org/versions/r0.10/get_started/os_setup.html) on your machine and execute the following from the root directory:

    python tests/callbacks.py

TF summaries are being written to `outputs/summaries`, and they can be inspected via this command:

    tensorboard --logdir=runs:output/summaries --port 6006 --reload_interval 5


