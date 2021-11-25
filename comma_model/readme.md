CommaAI super combo model training pipeline.

To do: 

1. Intial Image transformations. (Done)
2. Dataloader and train script. 
3. Loss functions and task balancing strategy.  
4. Arrange the code into specific dirs. 

Goals week: 
1. complete the  generalised training pipeline.
2. Complete the dataloader to incorporate initial image transformations.
3. Initial train on the generated gt.

Random things that were discussed in the meeting:
1. We dont have enough data for the training the model from scratch, so we are want to convert the existing onnx model--->torch and train some of the last the layers and fix the rest. 
2. The training pipeline will stay the same as planned. 
