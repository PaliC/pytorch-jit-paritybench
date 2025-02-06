
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_argmax_diag_embed_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_argmax_diag_embed_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (4*x0), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr0 + (1 + 4*x0), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr0 + (2 + 4*x0), xmask, eviction_policy='evict_last')
    tmp46 = tl.load(in_ptr0 + (3 + 4*x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = x0
    tmp3 = tmp1 == tmp2
    tmp4 = tl.full([1], -1000000000, tl.int64)
    tmp5 = tl.where(tmp3, tmp4, tmp1)
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tmp0 + tmp6
    tmp9 = tl.full([1], 1, tl.int64)
    tmp10 = tmp9 == tmp2
    tmp11 = tl.where(tmp10, tmp4, tmp1)
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tmp8 + tmp12
    tmp14 = tmp7 > tmp13
    tmp15 = tmp7 == tmp13
    tmp16 = tmp7 != tmp7
    tmp17 = tmp13 != tmp13
    tmp18 = tmp16 > tmp17
    tmp19 = tmp14 | tmp18
    tmp20 = tmp16 & tmp17
    tmp21 = tmp15 | tmp20
    tmp22 = tmp1 < tmp9
    tmp23 = tmp21 & tmp22
    tmp24 = tmp19 | tmp23
    tmp25 = tl.where(tmp24, tmp7, tmp13)
    tmp26 = tl.where(tmp24, tmp1, tmp9)
    tmp28 = tl.full([1], 2, tl.int64)
    tmp29 = tmp28 == tmp2
    tmp30 = tl.where(tmp29, tmp4, tmp1)
    tmp31 = tmp30.to(tl.float32)
    tmp32 = tmp27 + tmp31
    tmp33 = tmp25 > tmp32
    tmp34 = tmp25 == tmp32
    tmp35 = tmp25 != tmp25
    tmp36 = tmp32 != tmp32
    tmp37 = tmp35 > tmp36
    tmp38 = tmp33 | tmp37
    tmp39 = tmp35 & tmp36
    tmp40 = tmp34 | tmp39
    tmp41 = tmp26 < tmp28
    tmp42 = tmp40 & tmp41
    tmp43 = tmp38 | tmp42
    tmp44 = tl.where(tmp43, tmp25, tmp32)
    tmp45 = tl.where(tmp43, tmp26, tmp28)
    tmp47 = tl.full([1], 3, tl.int64)
    tmp48 = tmp47 == tmp2
    tmp49 = tl.where(tmp48, tmp4, tmp1)
    tmp50 = tmp49.to(tl.float32)
    tmp51 = tmp46 + tmp50
    tmp52 = tmp44 > tmp51
    tmp53 = tmp44 == tmp51
    tmp54 = tmp44 != tmp44
    tmp55 = tmp51 != tmp51
    tmp56 = tmp54 > tmp55
    tmp57 = tmp52 | tmp56
    tmp58 = tmp54 & tmp55
    tmp59 = tmp53 | tmp58
    tmp60 = tmp45 < tmp47
    tmp61 = tmp59 & tmp60
    tmp62 = tmp57 | tmp61
    tmp63 = tl.where(tmp62, tmp44, tmp51)
    tmp64 = tl.where(tmp62, tmp45, tmp47)
    tl.store(out_ptr0 + (x0), tmp64, xmask)
