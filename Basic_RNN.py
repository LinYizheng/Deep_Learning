# RNN
# San Wong hswong1@uci.edu


'''
http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/
http://r2rt.com/recurrent-neural-networks-in-tensorflow-i.html


RNN is for sequential information. Assume all input and output are independent of each others


    ____ Output
   |    |
   |____|
     /|\
      |
      | V
    ____ Hidden State(i.e: memory) (Loop to itself with vector W)
   |    |
   |____|
     /|\
      |
      | U
    ____ Input
   |    |
   |____|

S_t = f(U*X_t + W*S_(t-1)) where f is tanh or ReLU
O_t = softmax(V*S_t)

S_t contain the information in previous time (but not all). O_t only calculate solely based on memory at time t
RNN use the same U V W across all steps -> this can reduces the total amount of parameteras

RNN also use BP. As all the weight matrix doesn't change over time. What changes is time steps
Therefore BP is actually Backpropagation Through Time (BPTT)
BPTT isn't easy learning Long Term dependencies
LSTM aims to solve this issue

# Diff Types of RNN
(1) Bidirectional RNNs
* Output at time t may not only depends on the previous elements in the sequence. It can also depends on Future elements
Therefore, you want to look at both (pass) and (future)

(2) Deep (Bidirectional) RNNs
On top of Bidirectional RNNs, there are multiple layers per time step. It gives higher learning capacity but also require a lot of training data

(3) LSTM
* Use diff function to compute the hidden state
* Memory in LSTMs are called cell, which take 2 inputs (1: Previous state h_(t-1) and 2: Current Input x_t)
* Cell decides what to keep in and what to forget
* Good for Long term depend data




time step: t : from 0 to n
Input = Binary vector: X_t (i.e: Either 0 or 1)
Generate a State Vector: S_t
Output =  Predicted Probability Distribution Vector P_t for Binary Vector Y_t

S_t = tanh(W(X_t@S_t-1 ) + b_s)  @ is Vector concatenation
P_t = softmax(US_t + b_p)

X_t is from R^2
W is from R^(d*(d+2))
b_s is from R^d
U is from R^2*d
b_p is from R^2

d is set to be 4


At t = 0, S_(-1) is set to 0

'''