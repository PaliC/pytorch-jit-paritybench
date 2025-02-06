
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_copy_mean_mul_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 32, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_copy_mean_mul_0(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 16)
    x1 = xindex // 16
    x4 = xindex
    x2 = (xindex % 4)
    x3 = ((xindex // 4) % 4)
    tmp0 = tl.load(in_ptr0 + (x0 + 64*x1), xmask)
    tmp1 = tl.load(in_ptr1 + (x0 + 64*x1), xmask)
    tmp3 = tl.load(in_ptr0 + (16 + x0 + 64*x1), xmask)
    tmp4 = tl.load(in_ptr1 + (16 + x0 + 64*x1), xmask)
    tmp7 = tl.load(in_ptr0 + (32 + x0 + 64*x1), xmask)
    tmp8 = tl.load(in_ptr1 + (32 + x0 + 64*x1), xmask)
    tmp11 = tl.load(in_ptr0 + (48 + x0 + 64*x1), xmask)
    tmp12 = tl.load(in_ptr1 + (48 + x0 + 64*x1), xmask)
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 * tmp4
    tmp6 = tmp2 + tmp5
    tmp9 = tmp7 * tmp8
    tmp10 = tmp6 + tmp9
    tmp13 = tmp11 * tmp12
    tmp14 = tmp10 + tmp13
    tmp15 = 4.0
    tmp16 = tmp14 / tmp15
    tmp17 = x2
    tmp18 = tl.full([1], 1, tl.int64)
    tmp19 = tmp17 >= tmp18
    tmp20 = tl.load(in_ptr0 + (x0 + 64*x1), tmp19 & xmask, other=0.0)
    tmp21 = tl.load(in_ptr1 + ((-1) + x0 + 64*x1), tmp19 & xmask, other=0.0)
    tmp22 = tmp20 * tmp21
    tmp23 = tl.load(in_ptr0 + (16 + x0 + 64*x1), tmp19 & xmask, other=0.0)
    tmp24 = tl.load(in_ptr1 + (15 + x0 + 64*x1), tmp19 & xmask, other=0.0)
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = tl.load(in_ptr0 + (32 + x0 + 64*x1), tmp19 & xmask, other=0.0)
    tmp28 = tl.load(in_ptr1 + (31 + x0 + 64*x1), tmp19 & xmask, other=0.0)
    tmp29 = tmp27 * tmp28
    tmp30 = tmp26 + tmp29
    tmp31 = tl.load(in_ptr0 + (48 + x0 + 64*x1), tmp19 & xmask, other=0.0)
    tmp32 = tl.load(in_ptr1 + (47 + x0 + 64*x1), tmp19 & xmask, other=0.0)
    tmp33 = tmp31 * tmp32
    tmp34 = tmp30 + tmp33
    tmp35 = 4.0
    tmp36 = tmp34 / tmp35
    tmp37 = tl.full(tmp36.shape, 0.0, tmp36.dtype)
    tmp38 = tl.where(tmp19, tmp36, tmp37)
    tmp39 = tl.full([1], 1, tl.int32)
    tmp40 = tl.full([1], 0, tl.int32)
    tmp41 = tmp39 == tmp40
    tmp42 = 0.0
    tmp43 = tl.where(tmp41, tmp16, tmp42)
    tmp44 = tl.where(tmp19, tmp38, tmp43)
    tmp45 = tl.full([1], 2, tl.int64)
    tmp46 = tmp17 >= tmp45
    tmp47 = tl.load(in_ptr0 + (x0 + 64*x1), tmp46 & xmask, other=0.0)
    tmp48 = tl.load(in_ptr1 + ((-2) + x0 + 64*x1), tmp46 & xmask, other=0.0)
    tmp49 = tmp47 * tmp48
    tmp50 = tl.load(in_ptr0 + (16 + x0 + 64*x1), tmp46 & xmask, other=0.0)
    tmp51 = tl.load(in_ptr1 + (14 + x0 + 64*x1), tmp46 & xmask, other=0.0)
    tmp52 = tmp50 * tmp51
    tmp53 = tmp49 + tmp52
    tmp54 = tl.load(in_ptr0 + (32 + x0 + 64*x1), tmp46 & xmask, other=0.0)
    tmp55 = tl.load(in_ptr1 + (30 + x0 + 64*x1), tmp46 & xmask, other=0.0)
    tmp56 = tmp54 * tmp55
    tmp57 = tmp53 + tmp56
    tmp58 = tl.load(in_ptr0 + (48 + x0 + 64*x1), tmp46 & xmask, other=0.0)
    tmp59 = tl.load(in_ptr1 + (46 + x0 + 64*x1), tmp46 & xmask, other=0.0)
    tmp60 = tmp58 * tmp59
    tmp61 = tmp57 + tmp60
    tmp62 = 4.0
    tmp63 = tmp61 / tmp62
    tmp64 = tl.full(tmp63.shape, 0.0, tmp63.dtype)
    tmp65 = tl.where(tmp46, tmp63, tmp64)
    tmp66 = tl.full([1], 2, tl.int32)
    tmp67 = tmp66 == tmp39
    tmp68 = tmp66 == tmp40
    tmp69 = tl.where(tmp68, tmp16, tmp42)
    tmp70 = tl.where(tmp67, tmp44, tmp69)
    tmp71 = tl.where(tmp46, tmp65, tmp70)
    tmp72 = tl.full([1], 3, tl.int64)
    tmp73 = tmp17 >= tmp72
    tmp74 = tl.load(in_ptr0 + (3 + 4*x3 + 64*x1), tmp73 & xmask, eviction_policy='evict_last', other=0.0)
    tmp75 = tl.load(in_ptr1 + (4*x3 + 64*x1), tmp73 & xmask, eviction_policy='evict_last', other=0.0)
    tmp76 = tmp74 * tmp75
    tmp77 = tl.load(in_ptr0 + (19 + 4*x3 + 64*x1), tmp73 & xmask, eviction_policy='evict_last', other=0.0)
    tmp78 = tl.load(in_ptr1 + (16 + 4*x3 + 64*x1), tmp73 & xmask, eviction_policy='evict_last', other=0.0)
    tmp79 = tmp77 * tmp78
    tmp80 = tmp76 + tmp79
    tmp81 = tl.load(in_ptr0 + (35 + 4*x3 + 64*x1), tmp73 & xmask, eviction_policy='evict_last', other=0.0)
    tmp82 = tl.load(in_ptr1 + (32 + 4*x3 + 64*x1), tmp73 & xmask, eviction_policy='evict_last', other=0.0)
    tmp83 = tmp81 * tmp82
    tmp84 = tmp80 + tmp83
    tmp85 = tl.load(in_ptr0 + (51 + 4*x3 + 64*x1), tmp73 & xmask, eviction_policy='evict_last', other=0.0)
    tmp86 = tl.load(in_ptr1 + (48 + 4*x3 + 64*x1), tmp73 & xmask, eviction_policy='evict_last', other=0.0)
    tmp87 = tmp85 * tmp86
    tmp88 = tmp84 + tmp87
    tmp89 = 4.0
    tmp90 = tmp88 / tmp89
    tmp91 = tl.full(tmp90.shape, 0.0, tmp90.dtype)
    tmp92 = tl.where(tmp73, tmp90, tmp91)
    tmp93 = tl.full([1], 3, tl.int32)
    tmp94 = tmp93 == tmp66
    tmp95 = tmp93 == tmp39
    tmp96 = tmp93 == tmp40
    tmp97 = tl.where(tmp96, tmp16, tmp42)
    tmp98 = tl.where(tmp95, tmp44, tmp97)
    tmp99 = tl.where(tmp94, tmp71, tmp98)
    tmp100 = tl.where(tmp73, tmp92, tmp99)
    tl.store(out_ptr0 + (x4), tmp16, xmask)
    tl.store(out_ptr1 + (x4), tmp44, xmask)
    tl.store(out_ptr2 + (x4), tmp71, xmask)
    tl.store(out_ptr3 + (x4), tmp100, xmask)
