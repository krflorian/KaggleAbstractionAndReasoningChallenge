Pretrain task:
1. Rotation - predict angle/direction `pretrain:done`
2. Remove row/column - predict row or column and line number `pretrain:done`
3. Shift line/column - predict row or column and line number `pretrain:done`
3. Change color of non background - predict which color changed from what to which `pretrain:done` (bit different then here)
4. Expand all grids to 30 x 30 - predict size of target
5. Predict how many colors changed in target
6. Predict ratio of figure/background in target 
7. Predict how many "objects" are in input and output
8. Multiply img - predict how often it's multiplied

Idea:
1. Model Input = (input, output)
   1. take output from other input sometimes
   2. model has to predict if output belongs to input or not


OPEN QUESTIONS:
1. is our input good?
   1. Should we change to single task input?
   2. Benefits would be much more train data - since we can use every picture for pretrain tasks
   3. Shortcomings would be: how should the model learn to predict size?
   4. But how is this currently handled?

Questions:

Ideas:
1. Two model approach (propably to little data)
	- train one model to create metadata which will be used by another model
2. Use this pretraintask to create meta data model


https://arxiv.org/pdf/1907.11879.pdf

Use this approach like in the paper.

Create a big model which splits up at the end to perform the pretaintasks.
Only the last few layers will be split up for it's specific tasks.
Take this middle network and freeze it. Add additional layers to fine tune for the wanted output.
