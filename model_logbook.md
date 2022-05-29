# Model Logbook

## Goal:

  Get subtitles from continuous videos of sign language sentences


## Purpose of this document:

	To provide a source of truth of what we did, when, and why, and any context that was important to those
	decisions.

# Table of contents

- [Model Logbook](#model-logbook)
  - [Goal:](#goal)
  - [Purpose of this document:](#purpose-of-this-document)
- [Table of contents](#table-of-contents)
  - [Methodology](#methodology)
  - [TODOs](#todos)
  - [Instructions](#instructions)
  - [Dated logs](#dated-logs)
    - [\[29/05\] CNN: Simple CNN model](#2905-cnn-simple-cnn-model)


## Methodology

1. Make code work
2. Make model overfit a small sample
   1. Proves architecture can train
   2. Intuition on HP
3. Find a good set of HP to overfit on the complete dataset
   1. Proves architecture can train from data
   2. Good set of HP
4. Make the model generalize on validation set


## TODOs

| Value | Time (H) | Label  |                                                                                                   Description | State |
| :---- | :------- | :----: | ------------------------------------------------------------------------------------------------------------: | ----: |
| 75    | 1        | Doc    |                                                                                                       Logbook |   [x] |
| 70-80 | 4        | Model  |                                                                                                           ViT |   [ ] |
| 80    | 2        | Train  |                                                                                 Train simple cnn on one epoch |   [ ] |


## Instructions

Inspiration from [OPT-175B Logbook](https://github.com/facebookresearch/metaseq/blob/main/projects/OPT/chronicles/OPT175B_Logbook.pdf)

- Add a dated entry for each log, in reverse chronological order.
- Entries do not have to correspond to launches, but may include notes.
- For all launches, include:
  - Date
  - Context of why changes were necessary (Analysis of previous run)
    - Include tensorboard screenshots of spikes or divergences if applicable
  - Action:
    - If launch:
      - Launch steps:
      - Checkpoint/log folder/hparams
  - Results:
    - Include tensorboard screenshots of spikes or divergences if applicable
  - Next Steps:
    - From


## Dated logs


### [29/05] CNN: Simple CNN model

Context:
 - Test the baseline

Action:
 - Start a training with a small dataset
 - Batch size = 1
 - Command:
```sh
python train.py datamodule=how2sign.yaml model=simple_cnn.yaml
```
 - ckpt_path:
`./logs/experiments/runs/default/2022-05-29_11-29-49/checkpoints/epoch_000.ckpt`

Results:
 - Accuracy train: 0

Next Step:
 - Add ViT layer

[Table of contents](#table-of-contents)
