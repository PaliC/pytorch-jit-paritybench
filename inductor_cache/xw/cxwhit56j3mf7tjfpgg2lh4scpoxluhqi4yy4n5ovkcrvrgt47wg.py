
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_pow_sub_sum_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 16, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_pow_sub_sum_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 4)
    x1 = xindex // 4
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 64*x1), xmask)
    tmp1 = tl.load(in_ptr0 + (16 + x0 + 64*x1), xmask)
    tmp3 = tl.load(in_ptr0 + (32 + x0 + 64*x1), xmask)
    tmp5 = tl.load(in_ptr0 + (48 + x0 + 64*x1), xmask)
    tmp16 = tl.load(in_ptr0 + (4 + x0 + 64*x1), xmask)
    tmp17 = tl.load(in_ptr0 + (20 + x0 + 64*x1), xmask)
    tmp19 = tl.load(in_ptr0 + (36 + x0 + 64*x1), xmask)
    tmp21 = tl.load(in_ptr0 + (52 + x0 + 64*x1), xmask)
    tmp33 = tl.load(in_ptr0 + (8 + x0 + 64*x1), xmask)
    tmp34 = tl.load(in_ptr0 + (24 + x0 + 64*x1), xmask)
    tmp36 = tl.load(in_ptr0 + (40 + x0 + 64*x1), xmask)
    tmp38 = tl.load(in_ptr0 + (56 + x0 + 64*x1), xmask)
    tmp50 = tl.load(in_ptr0 + (12 + x0 + 64*x1), xmask)
    tmp51 = tl.load(in_ptr0 + (28 + x0 + 64*x1), xmask)
    tmp53 = tl.load(in_ptr0 + (44 + x0 + 64*x1), xmask)
    tmp55 = tl.load(in_ptr0 + (60 + x0 + 64*x1), xmask)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp7 = tmp6 * tmp6
    tmp8 = tmp0 * tmp0
    tmp9 = tmp1 * tmp1
    tmp10 = tmp8 + tmp9
    tmp11 = tmp3 * tmp3
    tmp12 = tmp10 + tmp11
    tmp13 = tmp5 * tmp5
    tmp14 = tmp12 + tmp13
    tmp15 = tmp7 - tmp14
    tmp18 = tmp16 + tmp17
    tmp20 = tmp18 + tmp19
    tmp22 = tmp20 + tmp21
    tmp23 = tmp22 * tmp22
    tmp24 = tmp16 * tmp16
    tmp25 = tmp17 * tmp17
    tmp26 = tmp24 + tmp25
    tmp27 = tmp19 * tmp19
    tmp28 = tmp26 + tmp27
    tmp29 = tmp21 * tmp21
    tmp30 = tmp28 + tmp29
    tmp31 = tmp23 - tmp30
    tmp32 = tmp15 + tmp31
    tmp35 = tmp33 + tmp34
    tmp37 = tmp35 + tmp36
    tmp39 = tmp37 + tmp38
    tmp40 = tmp39 * tmp39
    tmp41 = tmp33 * tmp33
    tmp42 = tmp34 * tmp34
    tmp43 = tmp41 + tmp42
    tmp44 = tmp36 * tmp36
    tmp45 = tmp43 + tmp44
    tmp46 = tmp38 * tmp38
    tmp47 = tmp45 + tmp46
    tmp48 = tmp40 - tmp47
    tmp49 = tmp32 + tmp48
    tmp52 = tmp50 + tmp51
    tmp54 = tmp52 + tmp53
    tmp56 = tmp54 + tmp55
    tmp57 = tmp56 * tmp56
    tmp58 = tmp50 * tmp50
    tmp59 = tmp51 * tmp51
    tmp60 = tmp58 + tmp59
    tmp61 = tmp53 * tmp53
    tmp62 = tmp60 + tmp61
    tmp63 = tmp55 * tmp55
    tmp64 = tmp62 + tmp63
    tmp65 = tmp57 - tmp64
    tmp66 = tmp49 + tmp65
    tmp67 = 0.5
    tmp68 = tmp66 * tmp67
    tl.store(in_out_ptr0 + (x2), tmp68, xmask)
