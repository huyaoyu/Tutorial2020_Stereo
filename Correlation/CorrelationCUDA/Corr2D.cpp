#include <torch/extension.h>

#include <vector>

// CUDA test interfaces.
torch::Tensor from_BCHW_2_BHWC_padded_cuda( torch::Tensor input, int padding );
torch::Tensor create_L(torch::Tensor r, const int kernelSize);

// CUDA module interfaces.
std::vector<torch::Tensor> corr_2d_forward_cuda( 
    torch::Tensor input0, torch::Tensor input1, 
    int padding, int kernelSize, int maxDisplacement, int strideK, int strideD );

std::vector<torch::Tensor> corr_2d_backward_cuda( torch::Tensor grad, torch::Tensor input0, torch::Tensor input1, 
    int padding, int kernelSize, int maxDisplacement, int strideK, int strideD );

std::vector<torch::Tensor> corr_2d_forward_zn_cuda( 
    torch::Tensor input0, torch::Tensor input1, 
    int padding, int kernelSize, int maxDisplacement, int strideK, int strideD );

std::vector<torch::Tensor> corr_2d_backward_zn_cuda( torch::Tensor grad, torch::Tensor input0, torch::Tensor input1, 
    torch::Tensor cr, torch::Tensor L0, torch::Tensor L1, 
    int padding, int kernelSize, int maxDisplacement, int strideK, int strideD );

// C++ interfaces.

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor. ")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous. ")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor test_from_BCHW_2_BHWC_padded(torch::Tensor input, int padding)
{
    CHECK_INPUT(input);

    return from_BCHW_2_BHWC_padded_cuda(input, padding);
}

torch::Tensor test_create_L(torch::Tensor r, const int kernelSize)
{
    CHECK_INPUT(r);

    return create_L(r, kernelSize);
}

std::vector<torch::Tensor> corr_2d_forward( 
    torch::Tensor input0, torch::Tensor input1, 
    int padding, int kernelSize, int maxDisplacement, int strideK, int strideD )
{
    CHECK_INPUT(input0);
    CHECK_INPUT(input1);

    return corr_2d_forward_cuda(input0, input1, 
        padding, kernelSize, maxDisplacement, strideK, strideD );
}

std::vector<torch::Tensor> corr_2d_backward( torch::Tensor grad,  
    torch::Tensor input0, torch::Tensor input1,  
    int padding, int kernelSize, int maxDisplacement, int strideK, int strideD )
{
    CHECK_INPUT(grad);
    CHECK_INPUT(input0);
    CHECK_INPUT(input1);

    return corr_2d_backward_cuda(grad, input0, input1, padding, kernelSize, maxDisplacement, strideK, strideD);
}

std::vector<torch::Tensor> corr_2d_forward_zn( 
    torch::Tensor input0, torch::Tensor input1, 
    int padding, int kernelSize, int maxDisplacement, int strideK, int strideD )
{
    CHECK_INPUT(input0);
    CHECK_INPUT(input1);

    return corr_2d_forward_zn_cuda(input0, input1, 
        padding, kernelSize, maxDisplacement, strideK, strideD );
}

std::vector<torch::Tensor> corr_2d_backward_zn( torch::Tensor grad,  
    torch::Tensor input0, torch::Tensor input1, 
    torch::Tensor cr, torch::Tensor L0, torch::Tensor L1, 
    int padding, int kernelSize, int maxDisplacement, int strideK, int strideD )
{
    CHECK_INPUT(grad);
    CHECK_INPUT(input0);
    CHECK_INPUT(input1);
    CHECK_INPUT(cr);
    CHECK_INPUT(L0);
    CHECK_INPUT(L1);

    return corr_2d_backward_zn_cuda(grad, input0, input1, cr, L0, L1, padding, kernelSize, maxDisplacement, strideK, strideD);
}

PYBIND11_MODULE( TORCH_EXTENSION_NAME, m )
{
    m.def("test_from_BCHW_2_BHWC_padded", &test_from_BCHW_2_BHWC_padded, "TF: test_from_BCHW_2_BHWC_padded. ");
    m.def("test_create_L", &test_create_L, "TF: test_create_L. ");
    m.def("forward", &corr_2d_forward, "Corr2D forward, CUDA version. ");
    m.def("backward", &corr_2d_backward, "Corr2D backward, CUDA version. ");
    m.def("forward_zn", &corr_2d_forward_zn, "Corr2D forward zero-normalized, CUDA version. ");
    m.def("backward_zn", &corr_2d_backward_zn, "Corr2D backward zero-normalized, CUDA version. ");
}

