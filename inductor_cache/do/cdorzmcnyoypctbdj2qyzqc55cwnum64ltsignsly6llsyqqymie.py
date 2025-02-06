
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4096}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_leaky_relu_mean_relu_13', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_leaky_relu_mean_relu_13(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 1024)
    x1 = xindex // 1024
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 4096*x1), None)
    tmp11 = tl.load(in_ptr0 + (1024 + x0 + 4096*x1), None)
    tmp20 = tl.load(in_ptr0 + (2048 + x0 + 4096*x1), None)
    tmp29 = tl.load(in_ptr0 + (3072 + x0 + 4096*x1), None)
    tmp1 = 0.0
    tmp2 = tmp0 > tmp1
    tmp3 = 0.01
    tmp4 = tmp0 * tmp3
    tmp5 = tl.where(tmp2, tmp0, tmp4)
    tmp6 = tl.full([1], 0, tl.int32)
    tmp7 = triton_helpers.maximum(tmp6, tmp5)
    tmp8 = triton_helpers.maximum(tmp6, tmp7)
    tmp9 = triton_helpers.maximum(tmp6, tmp8)
    tmp10 = triton_helpers.maximum(tmp6, tmp9)
    tmp12 = tmp11 > tmp1
    tmp13 = tmp11 * tmp3
    tmp14 = tl.where(tmp12, tmp11, tmp13)
    tmp15 = triton_helpers.maximum(tmp6, tmp14)
    tmp16 = triton_helpers.maximum(tmp6, tmp15)
    tmp17 = triton_helpers.maximum(tmp6, tmp16)
    tmp18 = triton_helpers.maximum(tmp6, tmp17)
    tmp19 = tmp10 + tmp18
    tmp21 = tmp20 > tmp1
    tmp22 = tmp20 * tmp3
    tmp23 = tl.where(tmp21, tmp20, tmp22)
    tmp24 = triton_helpers.maximum(tmp6, tmp23)
    tmp25 = triton_helpers.maximum(tmp6, tmp24)
    tmp26 = triton_helpers.maximum(tmp6, tmp25)
    tmp27 = triton_helpers.maximum(tmp6, tmp26)
    tmp28 = tmp19 + tmp27
    tmp30 = tmp29 > tmp1
    tmp31 = tmp29 * tmp3
    tmp32 = tl.where(tmp30, tmp29, tmp31)
    tmp33 = triton_helpers.maximum(tmp6, tmp32)
    tmp34 = triton_helpers.maximum(tmp6, tmp33)
    tmp35 = triton_helpers.maximum(tmp6, tmp34)
    tmp36 = triton_helpers.maximum(tmp6, tmp35)
    tmp37 = tmp28 + tmp36
    tmp38 = 4.0
    tmp39 = tmp37 / tmp38
    tl.store(out_ptr0 + (x2), tmp39, None)
