import torch
from customDtype import custom as Value

def test_sanity_check():

    x1 =Value(10)
    x2 = Value(0)
    w1 = Value(0)
    w2 = Value(0)
    x1w1 = x1*w1
    sig_x1w1 = x1w1.sigmoid()
    w2sig_x1w1 = sig_x1w1*w2
    a = w2sig_x1w1 + x2/2
    out = a.sigmoid()
    out.backprop()
    print(out.val,out.gradient)
    print(a.val,a.gradient)
    print(w2sig_x1w1.val,w2sig_x1w1.gradient)
    print(sig_x1w1.val,sig_x1w1.gradient)
    print(x1w1.val,x1w1.gradient)
    print(x1.gradient,x2.gradient,w1.gradient,w2.gradient)

    x = Value(2)
    out = x.sigmoid()
    out.backprop()
    print(out.val,out.gradient)
    print(x.val,x.gradient)

    x = Value(-4.0)
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backprop()
    xmg, ymg = x, y

    x = torch.Tensor([-4.0]).double()
    x.requires_grad = True
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    xpt, ypt = x, y

    # forward pass went well
    assert ymg.val == ypt.data.item()
    # backward pass went well
    assert xmg.gradient == xpt.grad.item()

def test_more_ops():

    a = Value(-4.0)
    b = Value(2.0)
    c = a + b
    d = a * b + b**3
    c += c + 1
    c += 1 + c + (-a)
    d += d * 2 + (b + a).relu()
    d += 3 * d + (b - a).tanh()
    e = c - d
    f = e**2
    g = f / 2.0
    g += 10.0 / f
    g.backprop()
    amg, bmg, gmg, fmg, emg, dmg, cmg = a, b, g, f, e, d, c

    a = torch.Tensor([-4.0]).double()
    b = torch.Tensor([2.0]).double()
    a.requires_grad = True
    b.requires_grad = True
    c = a + b
    d = a * b + b**3
    c = c + c + 1
    c = c + 1 + c + (-a)
    d = d + d * 2 + (b + a).relu()
    d = d + 3 * d + (b - a).tanh()
    e = c - d
    f = e**2
    g = f / 2.0
    g = g + 10.0 / f
    c.retain_grad()
    d.retain_grad()
    e.retain_grad()
    f.retain_grad()
    g.retain_grad()
    g.backward()
    apt, bpt, gpt, fpt, ept, dpt, cpt = a, b, g, f, e, d, c

    tol = 1e-6
    # forward pass went well
    print('a',amg.gradient,apt.grad.item())
    print('b',bmg.gradient,bpt.grad.item())
    print('g',gmg.gradient,gpt.grad.item())
    print('f',fmg.gradient,fpt.grad.item())
    print('e',emg.gradient,ept.grad.item())
    print('d',dmg.gradient,dpt.grad.item())
    print('c',cmg.gradient,cpt.grad.item())
    # backward pass went well
    assert abs(gmg.val - gpt.data.item()) < tol
    assert abs(bmg.gradient - bpt.grad.item()) < tol
    assert abs(amg.gradient - apt.grad.item()) < tol

if __name__ == '__main__':
    '''
    Function to verify the integrity of gradients calculated under various operations
    '''
    test_sanity_check()
    test_more_ops()
