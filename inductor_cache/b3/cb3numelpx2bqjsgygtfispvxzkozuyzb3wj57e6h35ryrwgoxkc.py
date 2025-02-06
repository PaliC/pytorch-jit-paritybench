
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i8', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_18', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_18(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 73728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = ((xindex // 4608) % 4)
    x1 = ((xindex // 256) % 18)
    x4 = (xindex % 4608)
    x5 = xindex // 4608
    x6 = xindex
    tmp0 = 2*x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 8, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + x1
    tmp7 = tmp6 >= tmp1
    tmp8 = tl.full([1], 17, tl.int64)
    tmp9 = tmp6 < tmp8
    tmp10 = tmp7 & tmp9
    tmp11 = tmp5 & tmp10
    tmp12 = tl.load(in_ptr0 + ((-256) + x4 + 8704*x5), tmp11, other=float("-inf"))
    tmp13 = x1
    tmp14 = tmp13 >= tmp1
    tmp15 = tmp13 < tmp8
    tmp16 = tmp14 & tmp15
    tmp17 = tmp5 & tmp16
    tmp18 = tl.load(in_ptr0 + (x4 + 8704*x5), tmp17, other=float("-inf"))
    tmp19 = triton_helpers.maximum(tmp18, tmp12)
    tmp20 = 1 + 2*x2
    tmp21 = tmp20 >= tmp1
    tmp22 = tmp20 < tmp3
    tmp23 = tmp21 & tmp22
    tmp24 = tmp23 & tmp10
    tmp25 = tl.load(in_ptr0 + (4096 + x4 + 8704*x5), tmp24, other=float("-inf"))
    tmp26 = triton_helpers.maximum(tmp25, tmp19)
    tmp27 = tmp23 & tmp16
    tmp28 = tl.load(in_ptr0 + (4352 + x4 + 8704*x5), tmp27, other=float("-inf"))
    tmp29 = triton_helpers.maximum(tmp28, tmp26)
    tmp30 = tmp18 > tmp12
    tmp31 = tl.full([1], 1, tl.int8)
    tmp32 = tl.full([1], 0, tl.int8)
    tmp33 = tl.where(tmp30, tmp31, tmp32)
    tmp34 = tmp25 > tmp19
    tmp35 = tl.full([1], 2, tl.int8)
    tmp36 = tl.where(tmp34, tmp35, tmp33)
    tmp37 = tmp28 > tmp26
    tmp38 = tl.full([1], 3, tl.int8)
    tmp39 = tl.where(tmp37, tmp38, tmp36)
    tl.store(out_ptr0 + (x6), tmp29, None)
    tl.store(out_ptr1 + (x6), tmp39, None)
