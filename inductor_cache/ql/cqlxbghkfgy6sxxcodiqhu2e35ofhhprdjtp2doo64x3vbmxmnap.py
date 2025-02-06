
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8192}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_3(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 16) % 100)
    x0 = (xindex % 16)
    x2 = xindex // 1600
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 4, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 16*(x1) + 64*x2), tmp4 & xmask, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 36, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + (x0 + 16*((-4) + x1) + 512*x2), tmp9 & xmask, other=0.0)
    tmp11 = tl.load(in_ptr2 + ((-4) + x1), tmp9 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp10 + tmp11
    tmp13 = 0.0
    tmp14 = tmp12 > tmp13
    tmp15 = 0.2
    tmp16 = tmp12 * tmp15
    tmp17 = tl.where(tmp14, tmp12, tmp16)
    tmp18 = tl.full(tmp17.shape, 0.0, tmp17.dtype)
    tmp19 = tl.where(tmp9, tmp17, tmp18)
    tmp20 = tmp0 >= tmp7
    tmp21 = tl.full([1], 68, tl.int64)
    tmp22 = tmp0 < tmp21
    tmp23 = tmp20 & tmp22
    tmp24 = tl.load(in_ptr3 + (x0 + 16*((-36) + x1) + 512*x2), tmp23 & xmask, other=0.0)
    tmp25 = tl.load(in_ptr4 + ((-36) + x1), tmp23 & xmask, eviction_policy='evict_last', other=0.0)
    tmp26 = tmp24 + tmp25
    tmp27 = 0.0
    tmp28 = tmp26 > tmp27
    tmp29 = 0.2
    tmp30 = tmp26 * tmp29
    tmp31 = tl.where(tmp28, tmp26, tmp30)
    tmp32 = tl.full(tmp31.shape, 0.0, tmp31.dtype)
    tmp33 = tl.where(tmp23, tmp31, tmp32)
    tmp34 = tmp0 >= tmp21
    tmp35 = tl.full([1], 100, tl.int64)
    tmp36 = tmp0 < tmp35
    tmp37 = tl.load(in_ptr5 + (x0 + 16*((-68) + x1) + 512*x2), tmp34 & xmask, other=0.0)
    tmp38 = tl.load(in_ptr6 + ((-68) + x1), tmp34 & xmask, eviction_policy='evict_last', other=0.0)
    tmp39 = tmp37 + tmp38
    tmp40 = 0.0
    tmp41 = tmp39 > tmp40
    tmp42 = 0.2
    tmp43 = tmp39 * tmp42
    tmp44 = tl.where(tmp41, tmp39, tmp43)
    tmp45 = tl.full(tmp44.shape, 0.0, tmp44.dtype)
    tmp46 = tl.where(tmp34, tmp44, tmp45)
    tmp47 = tl.where(tmp23, tmp33, tmp46)
    tmp48 = tl.where(tmp9, tmp19, tmp47)
    tmp49 = tl.where(tmp4, tmp5, tmp48)
    tl.store(out_ptr0 + (x3), tmp49, xmask)
