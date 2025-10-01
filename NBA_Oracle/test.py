import torch
#print(torch.backends.mps.is_available())  # True if supported
#print(torch.backends.mps.is_built())      # True if PyTorch built with MPS
import numpy as np

x = torch.rand(5, 3)
y = torch.rand(5, 3)
#print(x)
#print(y) 
z = torch.mul(x, y)
#print(z)

#torch.mul, torch.add, torch.sub, torch.div are element-wise operations for pytorch tensors

#print(x[:,0]) #first column
#print(x[1,0].item()) #single value as a python number

a = x.view(15) #reshaping tensor to 1D tensor with 15 elements
b = x.view(-1,15) #reshaping tensor to 2D tensor with 15 columns, inferring the number of rows
#print(b.size()) #torch.Size([1, 15])

d = np.ones(5)
e = torch.from_numpy(d) #convert numpy array to pytorch tensor
#print(e)

d += 1 #modify numpy array in place by adding 1 to each element, also modifies the tensor e since it shares the same memory
#print(e)

#checks if torch connects to apple GPU/MPS
if torch.mps.is_available():
    device = torch.device("mps")
    x = torch.ones(5, device=device)
    y = torch.ones(5)
    y = y.to(device)
    z = x + y
    #z = z.to("mps")
    #print(z)


x = torch.randn(3,requires_grad=True)
#print("x = ",x)
y = x + 2
#print("y = ", y)
z = y * y * 2
#z = z.mean()
#print("z = ", z)

v = torch.tensor([0.1,1.0,0.001],dtype=torch.float32)
z.backward(v) #dz/dx
#print(x.grad) #dz/dx printed

x = torch.randn(3,requires_grad=True)
#print(x)
y = x.detach()
#with torch.no_grad()
y = x + 2
#print(y)


#weights 
weights = torch.ones(4,requires_grad=True)

for epoch in range(3):
    model_output = (weights * 3).sum()

    model_output.backward()
    #print(weights.grad)
    weights.grad.zero_() #reset weights back to 0


#Forwards Pass = compute the loss
#Compute Local Gradients
#Backwards pass = compute dLoss/dWeights using chain rule

x = torch.tensor(1.0)
y = torch.tensor(2.0)
w = torch.tensor(1.0,requires_grad=True)

#forwards pass
y_hat = w * x
loss = (y_hat - y) ** 2

print(loss)

#backwards pass
loss.backward()
print(w.grad)

#update weights
