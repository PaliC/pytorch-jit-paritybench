
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32768}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_20', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_20(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25676
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 131)
    x3 = xindex // 6419
    x5 = ((xindex // 131) % 49)
    x2 = ((xindex // 917) % 7)
    x1 = ((xindex // 131) % 7)
    x6 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 130, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = x0
    tmp6 = tl.full([1], 0, tl.int64)
    tmp7 = tmp5 >= tmp6
    tmp8 = tl.full([1], 128, tl.int64)
    tmp9 = tmp5 < tmp8
    tmp10 = tmp9 & tmp4
    tmp11 = tl.load(in_ptr0 + (x5 + 49*(x0) + 6272*x3), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp5 >= tmp8
    tmp13 = tl.full([1], 129, tl.int64)
    tmp14 = tmp5 < tmp13
    tmp15 = tmp12 & tmp14
    tmp16 = tmp15 & tmp4
    tmp17 = x2
    tmp18 = tmp17.to(tl.float32)
    tmp19 = 0.16666666666666666
    tmp20 = tmp18 * tmp19
    tmp21 = 2.0
    tmp22 = tmp20 * tmp21
    tmp23 = 1.0
    tmp24 = tmp22 - tmp23
    tmp25 = tl.full(tmp24.shape, 0.0, tmp24.dtype)
    tmp26 = tl.where(tmp16, tmp24, tmp25)
    tmp27 = tmp5 >= tmp13
    tmp28 = tl.full([1], 130, tl.int64)
    tmp29 = tmp5 < tmp28
    tmp30 = tmp27 & tmp4
    tmp31 = x1
    tmp32 = tmp31.to(tl.float32)
    tmp33 = 0.16666666666666666
    tmp34 = tmp32 * tmp33
    tmp35 = 2.0
    tmp36 = tmp34 * tmp35
    tmp37 = 1.0
    tmp38 = tmp36 - tmp37
    tmp39 = tl.full(tmp38.shape, 0.0, tmp38.dtype)
    tmp40 = tl.where(tmp30, tmp38, tmp39)
    tmp41 = tl.where(tmp15, tmp26, tmp40)
    tmp42 = tl.where(tmp9, tmp11, tmp41)
    tmp43 = tl.full(tmp42.shape, 0.0, tmp42.dtype)
    tmp44 = tl.where(tmp4, tmp42, tmp43)
    tmp45 = tmp0 >= tmp3
    tmp46 = tl.full([1], 131, tl.int64)
    tmp47 = tmp0 < tmp46
    tmp48 = x2
    tmp49 = tmp48.to(tl.float32)
    tmp50 = 0.16666666666666666
    tmp51 = tmp49 * tmp50
    tmp52 = 2.0
    tmp53 = tmp51 * tmp52
    tmp54 = 1.0
    tmp55 = tmp53 - tmp54
    tmp56 = 0.5
    tmp57 = tmp55 - tmp56
    tmp58 = tmp57 * tmp57
    tmp59 = x1
    tmp60 = tmp59.to(tl.float32)
    tmp61 = tmp60 * tmp50
    tmp62 = tmp61 * tmp52
    tmp63 = tmp62 - tmp54
    tmp64 = tmp63 - tmp56
    tmp65 = tmp64 * tmp64
    tmp66 = tmp58 + tmp65
    tmp67 = libdevice.sqrt(tmp66)
    tmp68 = tl.full(tmp67.shape, 0.0, tmp67.dtype)
    tmp69 = tl.where(tmp45, tmp67, tmp68)
    tmp70 = tl.where(tmp4, tmp44, tmp69)
    tl.store(out_ptr0 + (x6), tmp70, xmask)
