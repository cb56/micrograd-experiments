# Micograd Experiments
This is a small project I made following Andrej Karpathy's micrograd tutorial found here: [Andrej Karpathy: building micrograd](https://github.com/karpathy/micrograd)

## Project structure
- micrograd
  - `autograd.py`: contains the `Value` class, which is used to build nodes and execute mathematical expressions
  - `nn.py`: contains the neural net classes `Neuron`, `Layer` and `MLP`
- tests
  - `test_autograd.py`: test coverage around certain methods found within the Value class
