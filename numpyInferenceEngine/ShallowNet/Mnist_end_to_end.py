import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt


np.set_printoptions(suppress=True)


# load the mnist dataset

def fetch(url):
    import requests, gzip, os, hashlib, numpy
    fp = os.path.join("/tmp", hashlib.md5(url.encode('utf-8')).hexdigest())
    if not os.path.isfile(fp):
        with open(fp, "wb") as f:
            dat = requests.get(url).content
            f.write(dat)
    else:
        with open(fp, "rb") as f:
            dat = f.read()    
    return numpy.frombuffer(gzip.decompress(dat), dtype=np.uint8).copy()



X_train = fetch("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28, 28))
Y_train = fetch("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz")[8:]
X_test = fetch("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28, 28))
Y_test = fetch("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz")[8:]


import torch
import torch.nn as nn



class ShallowNet(torch.nn.Module):
    def __init__(self):
        super(ShallowNet, self).__init__()
        self.l1 = nn.Linear(784, 128, bias=False)
        self.act = nn.ReLU()
        self.l2 = nn.Linear(128, 10, bias=False)
        
    def forward(self, x):
        x = self.l1(x)
        x = self.act(x)
        x = self.l2(x)
        return x
    
model = ShallowNet()


BS = 128
loss_function = nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(), lr=0.001) # adam optimizer
losses, accuracies = [], []

for i in (t := trange(1000)):
    samp = np.random.randint(0, X_train.shape[0], size=(BS))
    X = torch.tensor(X_train[samp].reshape((-1, 28*28))).float()
    Y = torch.tensor(Y_train[samp]).long()
    optim.zero_grad()
    out = model(X)
    # compute accuracy
    cat = torch.argmax(out, dim=1)
    accuracy = (cat == Y).float().mean()
    loss = loss_function(out, Y)
    loss.backward()
    optim.step()
    losses.append(loss.item())
    accuracies.append(accuracy.item())
    t.set_description("loss %.2f accuracy %.2f" % (loss.item(), accuracy.item()))

plt.ylim(-0.1, 1.1)
plt.plot(losses)
plt.plot(accuracies)


torch.save(model.state_dict(), "weights.pkl")


#evaluation
Y_test_preds = torch.argmax(model(torch.tensor(X_test.reshape((-1, 28*28))).float()), dim=1).numpy()
print((Y_test_preds == Y_test).mean())

# copy weights from pytorch
l1 = model.l1.weight.detach().numpy().T
l2 = model.l2.weight.detach().numpy().T

# numpy forward pass
def forward(x):
    x = x.dot(l1)
    x = np.maximum(x, 0)
    x = x.dot(l2)
    return x

# eval
Y_test_preds_out = forward(X_test.reshape((-1, 28*28)))
Y_test_preds = np.argmax(Y_test_preds_out, axis=1)
print((Y_test == Y_test_preds).mean())




