
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_div_maximum_minimum_mul_neg_sub_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 17, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_div_maximum_minimum_mul_neg_sub_1(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 16
    x0 = (xindex % 16)
    x2 = xindex
    tmp76 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp0 = tl.full([1], 1, tl.int64)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tmp0 < tmp0
    tmp4 = tl.load(in_ptr0 + (4*x1), tmp3 & xmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_ptr0 + (1 + 4*x1), tmp3 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = 0.5
    tmp7 = tmp5 * tmp6
    tmp8 = tmp4 - tmp7
    tmp9 = tl.full(tmp8.shape, 0.0, tmp8.dtype)
    tmp10 = tl.where(tmp3, tmp8, tmp9)
    tmp11 = tmp0 >= tmp0
    tmp12 = tl.full([1], 2, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tl.load(in_ptr0 + (4*x1), tmp11 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = tl.load(in_ptr0 + (1 + 4*x1), tmp11 & xmask, eviction_policy='evict_last', other=0.0)
    tmp16 = 0.5
    tmp17 = tmp15 * tmp16
    tmp18 = tmp14 + tmp17
    tmp19 = tl.full(tmp18.shape, 0.0, tmp18.dtype)
    tmp20 = tl.where(tmp11, tmp18, tmp19)
    tmp21 = tl.where(tmp3, tmp10, tmp20)
    tmp22 = tmp1 >= tmp1
    tmp23 = tmp1 < tmp0
    tmp24 = tl.load(in_ptr0 + (4*x1), tmp23 & xmask, eviction_policy='evict_last', other=0.0)
    tmp25 = tl.load(in_ptr0 + (1 + 4*x1), tmp23 & xmask, eviction_policy='evict_last', other=0.0)
    tmp26 = 0.5
    tmp27 = tmp25 * tmp26
    tmp28 = tmp24 - tmp27
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp23, tmp28, tmp29)
    tmp31 = tmp1 >= tmp0
    tmp32 = tmp1 < tmp12
    tmp33 = tl.load(in_ptr0 + (4*x1), tmp31 & xmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr0 + (1 + 4*x1), tmp31 & xmask, eviction_policy='evict_last', other=0.0)
    tmp35 = 0.5
    tmp36 = tmp34 * tmp35
    tmp37 = tmp33 + tmp36
    tmp38 = tl.full(tmp37.shape, 0.0, tmp37.dtype)
    tmp39 = tl.where(tmp31, tmp37, tmp38)
    tmp40 = tl.where(tmp23, tmp30, tmp39)
    tmp41 = tmp21 - tmp40
    tmp42 = tl.load(in_ptr1 + (4*x0), tmp3 & xmask, eviction_policy='evict_last', other=0.0)
    tmp43 = tl.load(in_ptr1 + (1 + 4*x0), tmp3 & xmask, eviction_policy='evict_last', other=0.0)
    tmp44 = tmp43 * tmp6
    tmp45 = tmp42 - tmp44
    tmp46 = tl.full(tmp45.shape, 0.0, tmp45.dtype)
    tmp47 = tl.where(tmp3, tmp45, tmp46)
    tmp48 = tl.load(in_ptr1 + (4*x0), tmp11 & xmask, eviction_policy='evict_last', other=0.0)
    tmp49 = tl.load(in_ptr1 + (1 + 4*x0), tmp11 & xmask, eviction_policy='evict_last', other=0.0)
    tmp50 = tmp49 * tmp16
    tmp51 = tmp48 + tmp50
    tmp52 = tl.full(tmp51.shape, 0.0, tmp51.dtype)
    tmp53 = tl.where(tmp11, tmp51, tmp52)
    tmp54 = tl.where(tmp3, tmp47, tmp53)
    tmp55 = tl.load(in_ptr1 + (4*x0), tmp23 & xmask, eviction_policy='evict_last', other=0.0)
    tmp56 = tl.load(in_ptr1 + (1 + 4*x0), tmp23 & xmask, eviction_policy='evict_last', other=0.0)
    tmp57 = tmp56 * tmp26
    tmp58 = tmp55 - tmp57
    tmp59 = tl.full(tmp58.shape, 0.0, tmp58.dtype)
    tmp60 = tl.where(tmp23, tmp58, tmp59)
    tmp61 = tl.load(in_ptr1 + (4*x0), tmp31 & xmask, eviction_policy='evict_last', other=0.0)
    tmp62 = tl.load(in_ptr1 + (1 + 4*x0), tmp31 & xmask, eviction_policy='evict_last', other=0.0)
    tmp63 = tmp62 * tmp35
    tmp64 = tmp61 + tmp63
    tmp65 = tl.full(tmp64.shape, 0.0, tmp64.dtype)
    tmp66 = tl.where(tmp31, tmp64, tmp65)
    tmp67 = tl.where(tmp23, tmp60, tmp66)
    tmp68 = tmp54 - tmp67
    tmp69 = tmp41 + tmp68
    tmp70 = triton_helpers.minimum(tmp21, tmp54)
    tmp71 = triton_helpers.maximum(tmp40, tmp67)
    tmp72 = tmp70 - tmp71
    tmp73 = triton_helpers.maximum(tmp21, tmp54)
    tmp74 = triton_helpers.minimum(tmp40, tmp67)
    tmp75 = tmp73 - tmp74
    tmp77 = 1.0
    tmp78 = tmp76 * tmp77
    tmp79 = 0.0
    tmp80 = triton_helpers.maximum(tmp72, tmp79)
    tmp81 = tmp69 - tmp80
    tmp82 = tmp80 / tmp81
    tmp83 = triton_helpers.maximum(tmp75, tmp79)
    tmp84 = tmp83 - tmp81
    tmp85 = tmp84 / tmp83
    tmp86 = tmp82 - tmp85
    tmp87 = -tmp86
    tmp88 = tmp87 * tmp77
    tmp89 = tmp78 + tmp88
    tl.store(in_out_ptr0 + (x2), tmp89, xmask)
