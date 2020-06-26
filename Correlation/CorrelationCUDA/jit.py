from torch.utils.cpp_extension import load

DispCorrCUDA = load( 
    name="Corr2D_ext", 
    sources=[ "Corr2D.cpp", "Corr2D_Kernel.cu"]
 )

help(DispCorrCUDA)
