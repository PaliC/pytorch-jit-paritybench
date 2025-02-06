
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_25', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_25(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 4) % 4)
    x0 = (xindex % 4)
    x5 = xindex // 16
    x2 = ((xindex // 16) % 4)
    x3 = xindex // 64
    x6 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = x0
    tmp6 = tl.full([1], 0, tl.int64)
    tmp7 = tmp5 >= tmp6
    tmp8 = tl.full([1], 1, tl.int64)
    tmp9 = tmp5 < tmp8
    tmp10 = tmp9 & tmp4
    tmp11 = tl.load(in_ptr0 + (x5), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp5 >= tmp8
    tmp13 = tl.full([1], 2, tl.int64)
    tmp14 = tmp5 < tmp13
    tmp15 = tmp12 & tmp14
    tmp16 = tmp15 & tmp4
    tmp17 = tl.load(in_ptr1 + (x5), tmp16 & xmask, eviction_policy='evict_last', other=0.0)
    tmp18 = tmp5 >= tmp13
    tmp19 = tl.full([1], 4, tl.int64)
    tmp20 = tmp5 < tmp19
    tmp21 = tmp18 & tmp4
    tmp22 = tl.load(in_ptr2 + (x2 + 4*((-2) + x0) + 8*x3), tmp21 & xmask, eviction_policy='evict_last', other=0.0)
    tmp23 = tl.where(tmp15, tmp17, tmp22)
    tmp24 = tl.where(tmp9, tmp11, tmp23)
    tmp25 = tl.full(tmp24.shape, 0.0, tmp24.dtype)
    tmp26 = tl.where(tmp4, tmp24, tmp25)
    tmp27 = tmp0 >= tmp3
    tmp28 = tl.full([1], 2, tl.int64)
    tmp29 = tmp0 < tmp28
    tmp30 = tmp27 & tmp29
    tmp31 = x0
    tmp32 = tl.full([1], 0, tl.int64)
    tmp33 = tmp31 >= tmp32
    tmp34 = tl.full([1], 1, tl.int64)
    tmp35 = tmp31 < tmp34
    tmp36 = tmp35 & tmp30
    tmp37 = tl.load(in_ptr3 + (x5), tmp36 & xmask, eviction_policy='evict_last', other=0.0)
    tmp38 = tmp31 >= tmp34
    tmp39 = tl.full([1], 2, tl.int64)
    tmp40 = tmp31 < tmp39
    tmp41 = tmp38 & tmp40
    tmp42 = tmp41 & tmp30
    tmp43 = tl.load(in_ptr4 + (x5), tmp42 & xmask, eviction_policy='evict_last', other=0.0)
    tmp44 = tmp31 >= tmp39
    tmp45 = tl.full([1], 4, tl.int64)
    tmp46 = tmp31 < tmp45
    tmp47 = tmp44 & tmp30
    tmp48 = tl.load(in_ptr5 + (x2 + 4*((-2) + x0) + 8*x3), tmp47 & xmask, eviction_policy='evict_last', other=0.0)
    tmp49 = tl.where(tmp41, tmp43, tmp48)
    tmp50 = tl.where(tmp35, tmp37, tmp49)
    tmp51 = tl.full(tmp50.shape, 0.0, tmp50.dtype)
    tmp52 = tl.where(tmp30, tmp50, tmp51)
    tmp53 = tmp0 >= tmp28
    tmp54 = tl.full([1], 4, tl.int64)
    tmp55 = tmp0 < tmp54
    tmp56 = x0
    tmp57 = tl.full([1], 0, tl.int64)
    tmp58 = tmp56 >= tmp57
    tmp59 = tl.full([1], 1, tl.int64)
    tmp60 = tmp56 < tmp59
    tmp61 = tmp60 & tmp53
    tmp62 = tl.load(in_ptr6 + (x2 + 4*((-2) + x1) + 8*x3), tmp61 & xmask, eviction_policy='evict_last', other=0.0)
    tmp63 = tmp56 >= tmp59
    tmp64 = tl.full([1], 2, tl.int64)
    tmp65 = tmp56 < tmp64
    tmp66 = tmp63 & tmp65
    tmp67 = tmp66 & tmp53
    tmp68 = tl.load(in_ptr7 + (x2 + 4*((-2) + x1) + 8*x3), tmp67 & xmask, eviction_policy='evict_last', other=0.0)
    tmp69 = tmp56 >= tmp64
    tmp70 = tl.full([1], 4, tl.int64)
    tmp71 = tmp56 < tmp70
    tmp72 = tmp69 & tmp53
    tmp73 = tl.load(in_ptr8 + (x2 + 4*((-2) + x0) + 8*((-2) + x1) + 16*x3), tmp72 & xmask, eviction_policy='evict_last', other=0.0)
    tmp74 = tl.where(tmp66, tmp68, tmp73)
    tmp75 = tl.where(tmp60, tmp62, tmp74)
    tmp76 = tl.full(tmp75.shape, 0.0, tmp75.dtype)
    tmp77 = tl.where(tmp53, tmp75, tmp76)
    tmp78 = tl.where(tmp30, tmp52, tmp77)
    tmp79 = tl.where(tmp4, tmp26, tmp78)
    tl.store(out_ptr0 + (x6), tmp79, xmask)
