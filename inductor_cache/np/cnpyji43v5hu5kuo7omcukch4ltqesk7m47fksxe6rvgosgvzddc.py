
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 64}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_1(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = ((xindex // 4) % 4)
    x0 = (xindex % 2)
    x1 = ((xindex // 2) % 2)
    x3 = xindex // 16
    x4 = xindex
    tmp0 = x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 2, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 3*x1 + 9*(x2) + 18*x3), tmp4 & xmask, other=0.0)
    tmp6 = tl.load(in_ptr0 + (1 + x0 + 3*x1 + 9*(x2) + 18*x3), tmp4 & xmask, other=0.0)
    tmp7 = triton_helpers.maximum(tmp6, tmp5)
    tmp8 = tl.load(in_ptr0 + (3 + x0 + 3*x1 + 9*(x2) + 18*x3), tmp4 & xmask, other=0.0)
    tmp9 = triton_helpers.maximum(tmp8, tmp7)
    tmp10 = tl.load(in_ptr0 + (4 + x0 + 3*x1 + 9*(x2) + 18*x3), tmp4 & xmask, other=0.0)
    tmp11 = triton_helpers.maximum(tmp10, tmp9)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp4, tmp11, tmp12)
    tmp14 = tmp0 >= tmp3
    tmp15 = tl.full([1], 4, tl.int64)
    tmp16 = tmp0 < tmp15
    tmp17 = tl.load(in_ptr1 + (x0 + 3*x1 + 9*((-2) + x2) + 18*x3), tmp14 & xmask, other=0.0)
    tmp18 = tl.load(in_ptr1 + (1 + x0 + 3*x1 + 9*((-2) + x2) + 18*x3), tmp14 & xmask, other=0.0)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tmp20 = tl.load(in_ptr1 + (3 + x0 + 3*x1 + 9*((-2) + x2) + 18*x3), tmp14 & xmask, other=0.0)
    tmp21 = triton_helpers.maximum(tmp20, tmp19)
    tmp22 = tl.load(in_ptr1 + (4 + x0 + 3*x1 + 9*((-2) + x2) + 18*x3), tmp14 & xmask, other=0.0)
    tmp23 = triton_helpers.maximum(tmp22, tmp21)
    tmp24 = tl.full(tmp23.shape, 0.0, tmp23.dtype)
    tmp25 = tl.where(tmp14, tmp23, tmp24)
    tmp26 = tl.where(tmp4, tmp13, tmp25)
    tl.store(out_ptr0 + (x4), tmp26, xmask)
