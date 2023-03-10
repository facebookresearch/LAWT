# Source code for the papers Linear algebra with transformers and What is my math transformer doing?

This directory contains the source code for the two papers [Linear Algebra with Transformers](https://openreview.net/forum?id=Hp4g7FAXXG) (Transactions in Machine Learning Research, October 2022) (LAWT), and [What is my transformer doing?](https://arxiv.org/abs/2211.00170) (2nd Math AI Workshop at NeurIPS 2022) (WIMTD).


## Environment
* Requirements: Numpy, pyTorch, python 3.8+.
* OS: Tested on Linux Ubuntu, on Windows, add `--window true` to the command line.
* On a SLURM cluster, `--is_slurm_job true`. Multi-gpu training, which allows you to  increase your batch size by sharing t over several GPU requires a SLURM cluster.
* A NVIDIA/CUDA GPU is recommended of you intend to train models: if you do not have one, set `--cpu true` (and be very patient). CPU-only inference works fine.

## Running the programs
To run the program: `python train.py --dump_path MYPATH --exp_name EXPNAME --exp_id EXPID  --parameters (see below)`.

Training logs will be found in `MYPATH/EXPNAME/EXPID/train.log`, trained models will be `MYPATH/EXPNAME/EXPID/*.pth` . Please make MYPATH an absolute path: relative paths seem not to work on some systems. `--dump_path`and `--exp_name` are mandatory. If `--exp_id`is missing, the program will generate a random one. If `MYPATH/EXPNAME/EXPID`already exists, the program will reload the last saved model, and take it from there (i.e. relaunch an experiment).

To run inference/tests on a trained model : copy the runtime parameters from the corresponding `train.log` file, change the `exp_name` and `exp_id`, and set `--eval_only true` and `--reload_model MODELPATH` (the full path to the saved model). When testing out of distribution, adjust parameters accordingly (e.g. change `eigen_test_distribution`).

## Important Parameters

### Problem generation

`--operation` problem to be trained on: "transpose" (section 4.1), "matrix_sum" (sec. 4.2), "matrix_vector" (matrix vector product, sec. 4.3), "matrix_product" (full matrix product, sec. 4.3), "eigenvalues" (sec. 4.4), "eigenvectors" (sec. 4.5), "invert_matrix" (sec. 4.6), "syminverse" (inversion of symmetric matrices), "singularvalues" (sec. 4.7), "singularvectors" (SVD sec. 4.7), or "cotraining" to train on several tasks (appendix F.2)

`--cotraining` : the tasks to perform from "TADMEFI", see Appendix F.2

`--generator` matrices with random `uniform` or `gaussian` distributed coefficients

`--max_input_coeff` range of matrix coefficient

`--min_dimension` `--max_dimension` `--rectangular` matrix dimensions will be randomly generated from min to max_dimension (inclusive). Square matrices are generated, unless `rectangular`is true. Use `force_dimension` to generate rectangular matrices of a fixed size. Note: some problems only allow for square, or symmetric, matrices, see the problem descriptions in the paper.

`--eigen_distribution` eigenvalue distribution of the train set (semicircle, positive, uniform, gaussian, laplace, positive2, marcenko), for OOD experiments

`--eigen_test_distribution` main eigenvalue test distribution (semicircle, positive, uniform, gaussian, laplace, positive2, marcenko)

`--additional_test_distributions` comma separated list of additional test distributions (for the eigenvalue, eigenvector and symmetric inversion experiments)


### Matrix encoding

 `--output_encoding` `--input_encoding` : string of comma-separated parameters
* FP15, precision, max_exp
* float, precision, base_int
* floatsymbol, precision

precision in decimal places (significant digits - 1) for the 4 encodings from the paper, define:
* P10: "float,2,10"
* P1000: "float,2,1000"
* B1999: "floatsymbol,2"
* FP15: "FP15,2,16"

### Training and test loops

`--max_len` maximum length of input or output sequence

`--max_output_len` maximum length of output sequence, lower values make inference faster

`--eval_only` `--reload_model` only run evaluations from a trained model

`--eval_distrib` distributions to use at evaluation, for ood experiments

`--eval_verbose`set to one to export model prediction (in order to study failure cases)


### float16 / AMP API
`--fp16` use float16

`--amp` use amp for variable precision, -1: don't use, >=1 use in fp16, 0 use in fp32

### Transformer architecture (model size, depth and heads)
`--n_enc_layers` layers in the encoder

`--n_dec_layers` layers in the decoder

`--enc_emb_dim` dimensions in the encoder (the FFN hidden layer has 4 times this numbers)

`--dec_emb_dim` dimensions in the decoder (the FFN hidden layer has 4 times this numbers)

`--n_enc_heads` attention heads in the encoder (must divide `enc_emb_dim`)

`--n_dec_heads` attention heads in the decoder (must divide `dec_emb_dim`)



## A walk through the code base

`train.py` : the main program, argument parsing and main()

`src/slurm.py` `src/logger.py` `src/utils.py` : various utilities.

`src/trainer.py`: the training loop. Training uses teacher forcing.

`src/evaluator.py`: the test loop, run at the end of every epoch. Generation is auto-regressive.

`src/dataset.py`: the data loader.

`src/optim.py`: code for the various optimisers (on top of those defined by pyTorch, see get_optimizer() for a list). Redefines Adam, with warmup and two scheduling plans (InvSqrt and Cosine).

 `src/model/transformer.py`: the transformer code, initialized in `src/model/__init__.py`

 `src/envs/numeric.py`: problem-specific code, and arguments. Example generation is in gen_expr(), test-time evaluation of a transformer prediction in check_predictions().

 `src/envs/generators.py`: generation and evaluation routines for each task (addition, eigenvalues, inverse &c.). For each task, generate() is called by gen_expr() (generates a problem instance for this task), evaluate() by check_predictions() (verifies a model predition for this task).

 `src/envs/encoders.py`: matrix encoding and decoding. `numeric.py` calls encode() and decode(), which call write_float() and parse_float(). write_float() and parse_float() are overloaded for every encoding (P10, P1000, B1999, FP15)


## Reproducing the experiments

The code allows to reproduce all experiments in the papers, except those from appendix E from LAWT (alternative architectures).

The `sweeps` directory contains json files with the parameters used for main experiments in LAWT (transposition, addition, multiplication, eigenvalues, eigenvectors, inversion), and the out-of-distribution experiments from WIMTD (`ev_generators.json`, corresponding to seven experiments, one for each value of `--eigen_distribution`).

Trained models for different operations can be downloaded at the links below, each tarball contains a model (*.pth) a training log file, and a pickle file containing the parameters :
* [Transposition](https://dl.fbaipublicfiles.com/LAWT/transpose.tar.gz)
* [Matrix addition](https://dl.fbaipublicfiles.com/LAWT/add.tar.gz)
* [Matrix multiplication](https://dl.fbaipublicfiles.com/LAWT/product.tar.gz)
* [Eigenvalues, trained on semicircle matrices](https://dl.fbaipublicfiles.com/LAWT/eigenvalue_semicircle.tar.gz)
* [Eigenvalues, trained on Laplace matrices](https://dl.fbaipublicfiles.com/LAWT/eigenvalue_laplace.tar.gz`
* [Eigenvectors](https://dl.fbaipublicfiles.com/LAWT/eigenvectors.tar.gz)
* [Matrix inverse](https://dl.fbaipublicfiles.com/LAWT/inverse.tar.gz)

The data for the failure case analyses in WIMTD can be found in the (https://dl.fbaipublicfiles.com/LAWT/failures/) directory (files `eigenvectors_50k_9pc.gz`, `eigenvectors_5k_70pc`, `eigenvectors_50k_6x6.gz` and `inversion_50k.gz`). When unzipped in the `failures`directory, they can be read using the notebook in `notebooks`. They can also be reproduced by training models, using the parameters from `sweeps/eigenvectors.json` and `sweeps/invert.json`, or using the models uploaded on AWS (see above), and then testing the trained models on large datasets (setting `--eval_verbose 1`, `--eval_only true` and `--eval_size 50000` or larger).

## License - Community

LAWT is licensed, as per the license found in the LICENSE file.
See the [CONTRIBUTING](CONTRIBUTING.md) file for how to help out.

## References - citations

Linear Algebra with Transformers

`@article{
charton2022linear,
title={Linear algebra with transformers},
author={Charton, François},
journal={Transactions on Machine Learning Research},
year={2022},
url={https://openreview.net/forum?id=Hp4g7FAXXG},
}`

What is my math transformer doing? - Three experiments on explainability and generalization

`@misc{charton2022linear2,
  doi = {10.48550/ARXIV.2211.00170},
  url = {https://arxiv.org/abs/2211.00170},
  author = {Charton, François},
  keywords = {Machine Learning (cs.LG), Artificial Intelligence (cs.AI), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {What is my math transformer doing? -- Three results on interpretability and generalization},
  publisher = {arXiv},
  year = {2022}
}`
