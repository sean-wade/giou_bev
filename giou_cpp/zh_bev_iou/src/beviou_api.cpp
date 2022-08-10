#include <torch/serialize/tensor.h>
#include <torch/extension.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include "beviou_cpu.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("boxes_iou_bev_cpu", &boxes_iou_bev_cpu, "oriented boxes iou");
	m.def("boxes_giou_bev_cpu", &boxes_giou_bev_cpu, "oriented boxes giou");
}