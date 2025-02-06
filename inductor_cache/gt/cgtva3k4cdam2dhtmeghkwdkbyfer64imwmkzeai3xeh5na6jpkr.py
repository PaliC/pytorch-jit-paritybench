
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2048}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_cat_clone_copy_div_mul_sub_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_cat_clone_copy_div_mul_sub_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = ((xindex // 16) % 2)
    x0 = (xindex % 4)
    x1 = ((xindex // 4) % 4)
    x3 = xindex // 32
    x4 = xindex
    tmp0 = x2
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp3 = tl.full([1], 0, tl.int64)
    tmp4 = tl.full([1], 1, tl.int64)
    tmp5 = tmp3 < tmp4
    tmp6 = tl.full([1], 0, tl.int64)
    tmp7 = tmp6 >= tmp6
    tmp8 = tl.full([1], 1, tl.int64)
    tmp9 = tmp6 < tmp8
    tmp10 = tmp9 & tmp5
    tmp11 = x0
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tmp14 = tmp6 >= tmp8
    tmp15 = tl.full([1], 2, tl.int64)
    tmp16 = tmp6 < tmp15
    tmp17 = tmp14 & tmp5
    tmp18 = x1
    tmp19 = tl.full(tmp18.shape, 0.0, tmp18.dtype)
    tmp20 = tl.where(tmp17, tmp18, tmp19)
    tmp21 = tl.where(tmp9, tmp13, tmp20)
    tmp22 = tmp21.to(tl.float32)
    tmp23 = tl.load(in_ptr0 + (x0 + 4*((((x3 % 16)) % 4)) + 16*x1 + 64*(((x3 % 16)) // 4) + 256*(x3 // 16)), tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp24 = tmp22 - tmp23
    tmp25 = tl.full(tmp24.shape, 0.0, tmp24.dtype)
    tmp26 = tl.where(tmp5, tmp24, tmp25)
    tmp27 = tmp3 >= tmp3
    tmp28 = x0
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp5, tmp28, tmp29)
    tmp31 = tmp3 >= tmp4
    tmp32 = tl.full([1], 2, tl.int64)
    tmp33 = tmp3 < tmp32
    tmp34 = x1
    tmp35 = tl.full(tmp34.shape, 0.0, tmp34.dtype)
    tmp36 = tl.where(tmp31, tmp34, tmp35)
    tmp37 = tl.where(tmp5, tmp30, tmp36)
    tmp38 = tmp37.to(tl.float32)
    tmp39 = tl.where(tmp5, tmp26, tmp38)
    tmp40 = 2.0
    tmp41 = tmp39 * tmp40
    tmp42 = 0.3333333333333333
    tmp43 = tmp41 * tmp42
    tmp44 = 1.0
    tmp45 = tmp43 - tmp44
    tmp46 = tmp0 < tmp4
    tmp47 = x2
    tmp48 = tl.full([1], 0, tl.int64)
    tmp49 = tmp47 >= tmp48
    tmp50 = tl.full([1], 1, tl.int64)
    tmp51 = tmp47 < tmp50
    tmp52 = tmp51 & tmp46
    tmp53 = x0
    tmp54 = tl.full(tmp53.shape, 0.0, tmp53.dtype)
    tmp55 = tl.where(tmp52, tmp53, tmp54)
    tmp56 = tmp47 >= tmp50
    tmp57 = tl.full([1], 2, tl.int64)
    tmp58 = tmp47 < tmp57
    tmp59 = tmp56 & tmp46
    tmp60 = x1
    tmp61 = tl.full(tmp60.shape, 0.0, tmp60.dtype)
    tmp62 = tl.where(tmp59, tmp60, tmp61)
    tmp63 = tl.where(tmp51, tmp55, tmp62)
    tmp64 = tmp63.to(tl.float32)
    tmp65 = tl.load(in_ptr0 + (x0 + 4*((((x3 % 16)) % 4)) + 16*x1 + 64*(((x3 % 16)) // 4) + 256*(x3 // 16)), tmp46 & xmask, eviction_policy='evict_last', other=0.0)
    tmp66 = tmp64 - tmp65
    tmp67 = tl.full(tmp66.shape, 0.0, tmp66.dtype)
    tmp68 = tl.where(tmp46, tmp66, tmp67)
    tmp69 = tmp0 >= tmp3
    tmp70 = x0
    tmp71 = tl.full(tmp70.shape, 0.0, tmp70.dtype)
    tmp72 = tl.where(tmp46, tmp70, tmp71)
    tmp73 = tmp0 >= tmp4
    tmp74 = tmp0 < tmp32
    tmp75 = x1
    tmp76 = tl.full(tmp75.shape, 0.0, tmp75.dtype)
    tmp77 = tl.where(tmp73, tmp75, tmp76)
    tmp78 = tl.where(tmp46, tmp72, tmp77)
    tmp79 = tmp78.to(tl.float32)
    tmp80 = tl.where(tmp46, tmp68, tmp79)
    tmp81 = tl.where(tmp2, tmp45, tmp80)
    tl.store(out_ptr0 + (x4), tmp81, xmask)
