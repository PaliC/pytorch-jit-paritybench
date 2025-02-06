
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1024}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 4) % 12)
    x0 = (xindex % 4)
    x4 = xindex // 48
    x3 = xindex // 192
    x5 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 4, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 4*(x1) + 16*x4), tmp4 & xmask, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 8, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = x3
    tmp11 = tl.full([1], 1, tl.int64)
    tmp12 = tmp10 >= tmp11
    tmp13 = tmp12 & tmp9
    tmp14 = tl.load(in_ptr0 + (x0 + 4*((-4) + x1) + 16*x4), tmp13 & xmask, other=0.0)
    tmp15 = tl.load(in_ptr0 + ((-64) + x0 + 4*((-4) + x1) + 16*x4), tmp13 & xmask, other=0.0)
    tmp16 = tmp14 - tmp15
    tmp17 = tl.full(tmp16.shape, 0.0, tmp16.dtype)
    tmp18 = tl.where(tmp13, tmp16, tmp17)
    tmp19 = tl.load(in_ptr0 + (x0 + 4*((-4) + x1) + 16*x4), tmp9 & xmask, other=0.0)
    tmp20 = tl.where(tmp12, tmp18, tmp19)
    tmp21 = tl.full(tmp20.shape, 0.0, tmp20.dtype)
    tmp22 = tl.where(tmp9, tmp20, tmp21)
    tmp23 = tmp0 >= tmp7
    tmp24 = tl.full([1], 12, tl.int64)
    tmp25 = tmp0 < tmp24
    tmp26 = x3
    tmp27 = tl.full([1], 3, tl.int64)
    tmp28 = tmp26 < tmp27
    tmp29 = tmp28 & tmp23
    tmp30 = tl.load(in_ptr0 + (x0 + 4*((-8) + x1) + 16*x4), tmp29 & xmask, other=0.0)
    tmp31 = tl.load(in_ptr0 + (64 + x0 + 4*((-8) + x1) + 16*x4), tmp29 & xmask, other=0.0)
    tmp32 = tmp30 - tmp31
    tmp33 = tl.full(tmp32.shape, 0.0, tmp32.dtype)
    tmp34 = tl.where(tmp29, tmp32, tmp33)
    tmp35 = tl.load(in_ptr0 + (x0 + 4*((-8) + x1) + 16*x4), tmp23 & xmask, other=0.0)
    tmp36 = tl.where(tmp28, tmp34, tmp35)
    tmp37 = tl.full(tmp36.shape, 0.0, tmp36.dtype)
    tmp38 = tl.where(tmp23, tmp36, tmp37)
    tmp39 = tl.where(tmp9, tmp22, tmp38)
    tmp40 = tl.where(tmp4, tmp5, tmp39)
    tl.store(out_ptr0 + (x5), tmp40, xmask)
