import torch
import time

x = torch.randn(2000, 2000).cuda()
y = torch.randn(2000, 2000).cuda()



a = time.perf_counter()
out = torch.mm(x, y)
torch.cuda.synchronize()
b = time.perf_counter()
print(b-a)


# uncomment to take avarage of 200 ops
#total=0
#for i in range(0,200):
#    a = time.perf_counter()
#    out = torch.mm(x, y)
#    torch.cuda.synchronize()
#    b = time.perf_counter()
#    total+=(b-a)
#print(total/200)
