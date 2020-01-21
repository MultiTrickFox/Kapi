from torch import tensor, ones
from torch import sigmoid
from torch import pow, sum


inp = [
     [.2, .3 ,.1],
     [.1, .4, .7],
     [.1, .6, .4],
    ]

w1 = [
    [.1, .6 ,.2],
    [.7, -.2, -.1],
    [.2, .7, .1],
]

w2 = [
    [.3, .1 ,.01],
    [-.1, -.1, -.2],
    [-.3, .1, .2],
]

target = ones(3,3) * 3


inp = tensor(inp,requires_grad=True)
w1 = tensor(w1,requires_grad=True)
w2 = tensor(w2,requires_grad=True)
target = tensor(target)


out1 = inp @ w1
sum(out1).backward(retain_graph=True)
print(w1.grad)
w1.grad = None

out1 = sigmoid(out1)
sum(out1).backward(retain_graph=True)
print(w1.grad)
w1.grad = None

out2 = out1 @ w2
sum(out2).backward(retain_graph=True)
print(w2.grad)
w1.grad = None
w2.grad = None

loss = pow(target-out2,2)
sum(loss).backward()
print(w1.grad)


#print(loss)