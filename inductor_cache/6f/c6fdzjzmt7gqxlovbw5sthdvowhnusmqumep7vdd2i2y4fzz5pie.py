
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 512}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'in_ptr12': '*fp32', 'in_ptr13': '*fp32', 'in_ptr14': '*fp32', 'in_ptr15': '*fp32', 'in_ptr16': '*fp32', 'in_ptr17': '*fp32', 'in_ptr18': '*fp32', 'in_ptr19': '*fp32', 'in_ptr20': '*fp32', 'in_ptr21': '*fp32', 'in_ptr22': '*fp32', 'in_ptr23': '*fp32', 'in_ptr24': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_copy_zeros_8', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 25, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_copy_zeros_8(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, in_ptr21, in_ptr22, in_ptr23, in_ptr24, xnumel, XBLOCK : tl.constexpr):
    xnumel = 400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 4) % 25)
    x0 = (xindex % 4)
    x2 = xindex // 100
    x3 = xindex
    tmp3 = tl.load(in_ptr0 + (x0 + 4*x2), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr1 + (x0 + 4*x2), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr2 + (x0 + 4*x2), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0 + 4*x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x0 + 4*x2), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr5 + (x0 + 4*x2), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr6 + (x0 + 4*x2), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr7 + (x0 + 4*x2), xmask, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr8 + (x0 + 4*x2), xmask, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr9 + (x0 + 4*x2), xmask, eviction_policy='evict_last')
    tmp43 = tl.load(in_ptr10 + (x0 + 4*x2), xmask, eviction_policy='evict_last')
    tmp46 = tl.load(in_ptr11 + (x0 + 4*x2), xmask, eviction_policy='evict_last')
    tmp49 = tl.load(in_ptr12 + (x0 + 4*x2), xmask, eviction_policy='evict_last')
    tmp56 = tl.load(in_ptr13 + (x0 + 4*x2), xmask, eviction_policy='evict_last')
    tmp59 = tl.load(in_ptr14 + (x0 + 4*x2), xmask, eviction_policy='evict_last')
    tmp62 = tl.load(in_ptr15 + (x0 + 4*x2), xmask, eviction_policy='evict_last')
    tmp65 = tl.load(in_ptr16 + (x0 + 4*x2), xmask, eviction_policy='evict_last')
    tmp72 = tl.load(in_ptr17 + (x0 + 4*x2), xmask, eviction_policy='evict_last')
    tmp75 = tl.load(in_ptr18 + (x0 + 4*x2), xmask, eviction_policy='evict_last')
    tmp78 = tl.load(in_ptr19 + (x0 + 4*x2), xmask, eviction_policy='evict_last')
    tmp81 = tl.load(in_ptr20 + (x0 + 4*x2), xmask, eviction_policy='evict_last')
    tmp88 = tl.load(in_ptr21 + (x0 + 4*x2), xmask, eviction_policy='evict_last')
    tmp91 = tl.load(in_ptr22 + (x0 + 4*x2), xmask, eviction_policy='evict_last')
    tmp94 = tl.load(in_ptr23 + (x0 + 4*x2), xmask, eviction_policy='evict_last')
    tmp97 = tl.load(in_ptr24 + (x0 + 4*x2), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 4, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp4 = tl.full([1], 3, tl.int32)
    tmp5 = tmp0 == tmp4
    tmp7 = tl.full([1], 2, tl.int32)
    tmp8 = tmp0 == tmp7
    tmp10 = tl.full([1], 1, tl.int32)
    tmp11 = tmp0 == tmp10
    tmp13 = tl.full([1], 0, tl.int32)
    tmp14 = tmp0 == tmp13
    tmp16 = 0.0
    tmp17 = tl.where(tmp14, tmp15, tmp16)
    tmp18 = tl.where(tmp11, tmp12, tmp17)
    tmp19 = tl.where(tmp8, tmp9, tmp18)
    tmp20 = tl.where(tmp5, tmp6, tmp19)
    tmp21 = tl.where(tmp2, tmp3, tmp20)
    tmp22 = tl.full([1], 8, tl.int32)
    tmp23 = tmp0 == tmp22
    tmp25 = tl.full([1], 7, tl.int32)
    tmp26 = tmp0 == tmp25
    tmp28 = tl.full([1], 6, tl.int32)
    tmp29 = tmp0 == tmp28
    tmp31 = tl.full([1], 5, tl.int32)
    tmp32 = tmp0 == tmp31
    tmp34 = tl.where(tmp32, tmp33, tmp21)
    tmp35 = tl.where(tmp29, tmp30, tmp34)
    tmp36 = tl.where(tmp26, tmp27, tmp35)
    tmp37 = tl.where(tmp23, tmp24, tmp36)
    tmp38 = tl.full([1], 12, tl.int32)
    tmp39 = tmp0 == tmp38
    tmp41 = tl.full([1], 11, tl.int32)
    tmp42 = tmp0 == tmp41
    tmp44 = tl.full([1], 10, tl.int32)
    tmp45 = tmp0 == tmp44
    tmp47 = tl.full([1], 9, tl.int32)
    tmp48 = tmp0 == tmp47
    tmp50 = tl.where(tmp48, tmp49, tmp37)
    tmp51 = tl.where(tmp45, tmp46, tmp50)
    tmp52 = tl.where(tmp42, tmp43, tmp51)
    tmp53 = tl.where(tmp39, tmp40, tmp52)
    tmp54 = tl.full([1], 16, tl.int32)
    tmp55 = tmp0 == tmp54
    tmp57 = tl.full([1], 15, tl.int32)
    tmp58 = tmp0 == tmp57
    tmp60 = tl.full([1], 14, tl.int32)
    tmp61 = tmp0 == tmp60
    tmp63 = tl.full([1], 13, tl.int32)
    tmp64 = tmp0 == tmp63
    tmp66 = tl.where(tmp64, tmp65, tmp53)
    tmp67 = tl.where(tmp61, tmp62, tmp66)
    tmp68 = tl.where(tmp58, tmp59, tmp67)
    tmp69 = tl.where(tmp55, tmp56, tmp68)
    tmp70 = tl.full([1], 20, tl.int32)
    tmp71 = tmp0 == tmp70
    tmp73 = tl.full([1], 19, tl.int32)
    tmp74 = tmp0 == tmp73
    tmp76 = tl.full([1], 18, tl.int32)
    tmp77 = tmp0 == tmp76
    tmp79 = tl.full([1], 17, tl.int32)
    tmp80 = tmp0 == tmp79
    tmp82 = tl.where(tmp80, tmp81, tmp69)
    tmp83 = tl.where(tmp77, tmp78, tmp82)
    tmp84 = tl.where(tmp74, tmp75, tmp83)
    tmp85 = tl.where(tmp71, tmp72, tmp84)
    tmp86 = tl.full([1], 24, tl.int32)
    tmp87 = tmp0 == tmp86
    tmp89 = tl.full([1], 23, tl.int32)
    tmp90 = tmp0 == tmp89
    tmp92 = tl.full([1], 22, tl.int32)
    tmp93 = tmp0 == tmp92
    tmp95 = tl.full([1], 21, tl.int32)
    tmp96 = tmp0 == tmp95
    tmp98 = tl.where(tmp96, tmp97, tmp85)
    tmp99 = tl.where(tmp93, tmp94, tmp98)
    tmp100 = tl.where(tmp90, tmp91, tmp99)
    tmp101 = tl.where(tmp87, tmp88, tmp100)
    tl.store(in_out_ptr0 + (x3), tmp101, xmask)
