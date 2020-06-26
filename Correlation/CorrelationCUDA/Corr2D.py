import torch

import Corr2D_ext

def int_2_tensor(intList):
    return torch.tensor(intList, dtype=torch.int, requires_grad=False)

def tensor_2_int(t):
    assert len(t.size()) == 1
    assert t.size()[0] == 5
    assert t.dtype == torch.int

    return t.tolist()

class Corr2DF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x0, x1, maxDisplacement, \
            padding=1, kernelSize=3, strideK=1, strideD=1):

        ctx.maxDisplacement = maxDisplacement
        ctx.padding = padding
        ctx.kernelSize = kernelSize
        ctx.strideK = strideK
        ctx.strideD = strideD

        out = Corr2D_ext.forward(x0, x1, padding, kernelSize, maxDisplacement, strideK, strideD)

        ctx.save_for_backward(x0, x1)

        return out[0]

    @staticmethod
    def backward(ctx, grad):
        x0, x1 = ctx.saved_tensors

        output = Corr2D_ext.backward( grad, x0, x1,  
            ctx.padding, ctx.kernelSize, ctx.maxDisplacement, ctx.strideK, ctx.strideD )

        return output[0], output[1], None, None, None, None, None

class Corr2DM(torch.nn.Module):
    def __init__(self, maxDisplacement, padding=1, kernelSize=3, strideK=1, strideD=1):
        super(Corr2DM, self).__init__()

        assert maxDisplacement > 0
        assert kernelSize > 0
        assert kernelSize % 2 == 1
        assert strideK > 0
        assert strideD > 0

        self.maxDisplacement = maxDisplacement
        self.padding         = padding
        self.kernelSize      = kernelSize
        self.strideK         = strideK
        self.strideD         = strideD
    
    def forward(self, x0, x1):
        return Corr2DF.apply( x0, x1, self.maxDisplacement, \
            self.padding, self.kernelSize, self.strideK, self.strideD )

class Corr2DZNF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x0, x1, maxDisplacement, \
            padding=1, kernelSize=3, strideK=1, strideD=1):

        ctx.maxDisplacement = maxDisplacement
        ctx.padding = padding
        ctx.kernelSize = kernelSize
        ctx.strideK = strideK
        ctx.strideD = strideD

        out = Corr2D_ext.forward_zn(x0, x1, padding, kernelSize, maxDisplacement, strideK, strideD)

        ctx.save_for_backward(x0, x1, out[0], out[1], out[2])

        return out[0]

    @staticmethod
    def backward(ctx, grad):
        x0, x1, C, L0, L1 = ctx.saved_tensors

        output = Corr2D_ext.backward_zn( grad, x0, x1, C, L0, L1, 
            ctx.padding, ctx.kernelSize, ctx.maxDisplacement, ctx.strideK, ctx.strideD )

        return output[0], output[1], None, None, None, None, None

class Corr2DZNM(torch.nn.Module):
    def __init__(self, maxDisplacement, padding=1, kernelSize=3, strideK=1, strideD=1):
        super(Corr2DZNM, self).__init__()

        assert maxDisplacement > 0
        assert kernelSize > 0
        assert kernelSize % 2 == 1
        assert strideK > 0
        assert strideD > 0

        self.maxDisplacement = maxDisplacement
        self.padding         = padding
        self.kernelSize      = kernelSize
        self.strideK         = strideK
        self.strideD         = strideD
    
    def forward(self, x0, x1):
        return Corr2DZNF.apply( x0, x1, self.maxDisplacement, \
            self.padding, self.kernelSize, self.strideK, self.strideD )
