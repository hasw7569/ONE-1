import dynamic_tensor

model = Model()

model_input1_shape = [2, 3]   # first input shape of Sub. 12 float32s

dynamic_layer = dynamic_tensor.DynamicInputGenerator(model, model_input1_shape)

test_node_input = dynamic_layer.getTestNodeInput() # first input of Pow is dynamic tensor

i2 = Input("op2", "TENSOR_FLOAT32", "{2, 3}") # second input of Pow. 12 float32s
o1 = Output("op3", "TENSOR_FLOAT32", "{2, 3}")

model = model.Operation("POW", test_node_input, i2).To(o1) # Pow

model_input1_data = [1., 2., 3., 4., 5., 6.]
model_input2_data = [1., 2., 3., 0.5, 5., 2.]

input0 = {
      dynamic_layer.getModelInput(): model_input1_data,   # input 1
      dynamic_layer.getShapeInput() : model_input1_shape,

      i2: model_input2_data # input 2
      }

output0 = {
      o1: [1., 4., 27., 2., 3125., 36.]
           }

# Instantiate an example
Example((input0, output0))
