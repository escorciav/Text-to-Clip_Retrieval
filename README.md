# Preface

This is a functional fork of the [Text-to-Clip project](https://github.com/VisionLearningGroup/Text-to-Clip_Retrieval).

- We ease the installation and provide a docker environment to run this project.

- If this fork save you precious hours of painful installation, please cite our work:

  ```
  @article{EscorciaDJGS2018,
  author    = {Victor Escorcia and
               Cuong Duc Dao and
               Mihir Jain and
               Bernard Ghanem and
               Cees Snoek},
  title     = {Guess Where? Actor-Supervision for Spatiotemporal Action Localization},
  journal   = {CoRR},
  volume    = {abs/1804.01824},
  year      = {2018},
  url       = {http://arxiv.org/abs/1804.01824},
  archivePrefix = {arXiv},
  eprint    = {1804.01824}
  }
  ```

  TODO: update bibtex with Corpus-Moment-Retrieval-work

- If you like the installation procedure, give us a ⭐ in the github banner.

## Installation with Docker

1. Install [docker](https://docs.docker.com/install/) and [nvidia-docker](https://github.com/NVIDIA/nvidia-docker#quickstart).

    Please follow the installation instructions of your machine.
    As long as you can run docker hello-world container and test nvidia-smi with a cuda container, you are ready to go.

    - Let's test that you are ready by pulling out our docker image

        `docker run --runtime=nvidia -ti escorciavkaust/caffe-python-opencv:latest caffe device_query --gpu 0`

        You should read the information of your GPU in your terminal.

2. Let's go over the installation procedure without the headache of compilation errors.

    - Let's use a snapshot of the code with less headaches

        ```bash
        git clone git@github.com:escorciav/Text-to-Clip_Retrieval.git
        git checkout cp-functional-testing
        ````

    - Then, launch a container from the root folder of the project.

        `docker run --runtime=nvidia --rm -v /etc/passwd:/etc/passwd -u $(id -u):$(id -g) -v $(pwd):$(pwd) -w $(pwd) -ti escorciavkaust/caffe-python-opencv:latest bash`

    - In case, you are not in the working directory. Move to that folder.

        > Make sure that you replace the `[...]` with your filesystem structure when you copy-paste the command below 😅.

        `cd [...]/Text-to-Clip_Retrieval`

    - Follow the instructions outlined [here](#installation) from *step 2 onwards*.

That's all, two simple steps to get yourself up and running 😉.

### What if?

1. I close the container. Do I need to repeat the installation steps?

    Nope. All the libraries reside inside your root folder not in the image.

1. I close the container. How can I launch it again?

    Go to the root folder and type

    `docker run --runtime=nvidia --rm -v /etc/passwd:/etc/passwd -u $(id -u):$(id -g) -v $(pwd):$(pwd) -w $(pwd) -ti escorciavkaust/caffe-python-opencv:latest bash`

1. I wanna use pass my own data, and it is in a different folder. How can I access it from the container?

    Let's assume your data is in your `/scratch/awesome-dataset`

    `docker run --runtime=nvidia --rm -v /etc/passwd:/etc/passwd -u $(id -u):$(id -g) -v $(pwd):$(pwd) -v /scratch/awesome-dataset:/awesome-dataset -w $(pwd) -ti escorciavkaust/caffe-python-opencv:latest bash`

    You will find it in `/awesome-dataset` inside the container.

1. I can't find the `Text-to-Clip_Retrieval` folder inside the container.

    Most probably, you were not in the root folder when you launched it.

    > Make sure that you replace the `[...]` with your filesystem structure when you copy-paste the command below.

    ```bash
    cd [...]/Text-to-Clip_Retrieval
    docker run --runtime=nvidia --rm -v /etc/passwd:/etc/passwd -u $(id -u):$(id -g) -v $(pwd):$(pwd) -w $(pwd) -ti escorciavkaust/caffe-python-opencv:latest bash
    ```

1. Can you add the `Text-to-Clip` binaries to the docker image?

    Why not? gimme a ⭐ in the github banner and I will make time for that. The more stars I get, the priority increases.

## Organization details

This is a 3rdparty project, I used git to keep track different changes.

- The default branch `devel` corresponds to a derivative work related to [Corpus Moment Retrieval Project](https://github.com/escorciav/moments-retrieval).

  If this branch is useful for you, we would appreciate that you cite our work:

  ```
  @article{EscorciaDJGS2018,
  author    = {Victor Escorcia and
               Cuong Duc Dao and
               Mihir Jain and
               Bernard Ghanem and
               Cees Snoek},
  title     = {Guess Where? Actor-Supervision for Spatiotemporal Action Localization},
  journal   = {CoRR},
  volume    = {abs/1804.01824},
  year      = {2018},
  url       = {http://arxiv.org/abs/1804.01824},
  archivePrefix = {arXiv},
  eprint    = {1804.01824}
  }
  ```

  TODO: update bibtex with Corpus-Moment-Retrieval-work

- The original project is on the `master` branch.

- A functional version of the _Text-to-Clip project_ that is expected to run without issues is on the `cp-functional-testing` branch.

Original README 👇

---


# Multilevel Language and Vision Integration for Text-to-Clip Retrieval

Code released by Huijuan Xu (Boston University).

### Introduction

We address the problem of text-based activity retrieval in video. Given a
sentence describing an activity, our task is to retrieve matching clips
from an untrimmed video. Our model learns a fine-grained similarity metric
for retrieval and uses visual features to modulate the processing of query
sentences at the word level in a recurrent neural network. A multi-task
loss is also employed by adding query re-generation as an auxiliary task.


### License

Our code is released under the MIT License (refer to the LICENSE file for
details).

### Citing

If you find our paper useful in your research, please consider citing:


    @inproceedings{xu2019multilevel,
    title={Multilevel Language and Vision Integration for Text-to-Clip Retrieval.},
    author={Xu, Huijuan and He, Kun and Plummer, Bryan A. and Sigal, Leonid and Sclaroff,
    Stan and Saenko, Kate},
    booktitle={AAAI},
    year={2019}
    }


### Contents
1. [Installation](#installation)
2. [Preparation](#preparation)
3. [Train Proposal Network](#train_proposal_network)
4. [Extract Proposal Features](#extract_proposal_features)
5. [Training](#training)
6. [Testing](#testing)

### Installation:

1. Clone the Text-to-Clip_Retrieval repository.
   ```Shell
   git clone --recursive git@github.com:VisionLearningGroup/Text-to-Clip_Retrieval.git
   ```

2. Build `Caffe3d` with `pycaffe` (see: [Caffe installation
instructions](http://caffe.berkeleyvision.org/installation.html)).

   **Note:** Caffe must be built with Python support!

  ```Shell
  cd ./caffe3d

  # If have all of the requirements installed and your Makefile.config in
    place, then simply do:
  make -j8 && make pycaffe
  ```

3. Build lib folder.

   ```Shell
   cd ./lib
   make
   ```

### Preparation:

1. We convert the [orginal data annotation files](https://github.com/jiyanggao/TALL) into json format.

   ```Shell
   # train data json file
   caption_gt_train.json
   # test data json file
   caption_gt_test.json
   ```

2. Download the videos in [Charades
dataset](https://allenai.org/plato/charades/) and extract frames at 25fps.



### Train Proposal Network:

1. Generate the pickle data for training proposal network model.

   ```Shell
   cd ./preprocess
   # generate training data
   python generate_roidb_modified_freq1.py
   ```

2. Download C3D classification [pretrain model](https://drive.google.com/file/d/1os4a1K4pgjhRh8oiL7gO_DhM0NnCURHN/view?usp=sharing) to ./pretrain/ .

3. In root folder, run proposal network training:
   ```Shell
   bash ./experiments/train_rpn/script_train.sh
   ```

4. We provide one set of trained proposal network [model weights](https://drive.google.com/file/d/1w8TL-lm7wjOVTYgzBdHGvXbJc5AHZ16g/view?usp=sharing).


### Extract Proposal Features:

1. In root folder, extract proposal features for training data and save as
   hdf5 data.
   ```Shell
   bash ./experiments/extract_HDF_for_LSTM/script_test.sh
   ```


### Training:

1. In root folder, run:
   ```Shell
   bash ./experiments/Text_to_Clip/script_train.sh
   ```

### Testing:

1. Generate the pickle data for testing the Text_to_Clip model.

   ```Shell
   cd ./preprocess
   # generate test data
   python generate_roidb_modified_freq1_full_retrieval_test.py
   ```

2. Download one sample model to ./experiments/Text_to_Clip/snapshot/ .

   One Text_to_Clip model on Charades-STA dataset is provided in:
   [caffemodel
   .](https://drive.google.com/file/d/10C2gPLQXyNZ39CVLVKiWpYy5qu1xGvtX/view?usp=sharing)

   The provided model has Recall@1 (tIoU=0.7) score ~15.6% on the
   test set.

3. In root folder, generate the similarity scores on the test set and save
   as pickle file.
   ```Shell
   bash ./experiments/Text_to_Clip/test_fast/script_test.sh
   ```

4. Get the evaluation results.
   ```Shell
   cd ./experiments/Text_to_Clip/test_fast/evaluation/
   bash bash.sh
   ```

