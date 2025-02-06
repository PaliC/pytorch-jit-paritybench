
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 256}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'out_ptr5': '*fp32', 'out_ptr6': '*fp32', 'out_ptr7': '*fp32', 'out_ptr8': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 2, 6, 10, 11), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_cos_mul_sin_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_cos_mul_sin_0(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, out_ptr7, out_ptr8, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 4)
    x1 = xindex // 4
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (0))
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp6 = tl.load(in_ptr1 + (1))
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK])
    tmp11 = tl.load(in_ptr1 + (2))
    tmp12 = tl.broadcast_to(tmp11, [XBLOCK])
    tmp16 = tl.load(in_ptr1 + (3))
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK])
    tmp3 = tmp2 * tmp0
    tmp4 = tl_math.sin(tmp3)
    tmp5 = tl_math.cos(tmp3)
    tmp8 = tmp7 * tmp0
    tmp9 = tl_math.sin(tmp8)
    tmp10 = tl_math.cos(tmp8)
    tmp13 = tmp12 * tmp0
    tmp14 = tl_math.sin(tmp13)
    tmp15 = tl_math.cos(tmp13)
    tmp18 = tmp17 * tmp0
    tmp19 = tl_math.sin(tmp18)
    tmp20 = tl_math.cos(tmp18)
    tl.store(out_ptr0 + (x0 + 36*x1), tmp0, xmask)
    tl.store(out_ptr1 + (x0 + 36*x1), tmp4, xmask)
    tl.store(out_ptr2 + (x0 + 36*x1), tmp5, xmask)
    tl.store(out_ptr3 + (x0 + 36*x1), tmp9, xmask)
    tl.store(out_ptr4 + (x0 + 36*x1), tmp10, xmask)
    tl.store(out_ptr5 + (x0 + 36*x1), tmp14, xmask)
    tl.store(out_ptr6 + (x0 + 36*x1), tmp15, xmask)
    tl.store(out_ptr7 + (x0 + 36*x1), tmp19, xmask)
    tl.store(out_ptr8 + (x0 + 36*x1), tmp20, xmask)
