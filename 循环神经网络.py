import torch

batch_size = 1
seq_len = 3
input_size = 4
hidden_size = 2
num_layers = 1  # RNN层数

# Construction of RNN
rnn = torch.nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
# Wrapping the sequence into:(seqLen,batchSize,InputSize)
inputs = torch.randn(seq_len, batch_size, input_size)  # (3,1,4)
# Initializing the hidden to zero
hidden = torch.zeros(num_layers, batch_size, hidden_size)  # (1,1,2)

output, hidden = rnn(inputs, hidden)  # RNN内部包含了循环，故这里只需把整个序列输入即可

print('Output size:', output.shape)  # (seq_len, batch_size, hidden_size)
print('Output:', output)
print('Hidden size:', hidden.shape)  # (num_layers, batch_size, hidden_size)
print('Hidden:', hidden)

