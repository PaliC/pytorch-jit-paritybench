
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_13', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_13(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp3 = tl.load(in_ptr0 + (256 + x0), xmask)
    tmp6 = tl.load(in_ptr0 + (512 + x0), xmask)
    tmp9 = tl.load(in_ptr0 + (768 + x0), xmask)
    tmp1 = 1.0
    tmp2 = tmp0 / tmp1
    tmp4 = tmp3 / tmp1
    tmp5 = triton_helpers.maximum(tmp2, tmp4)
    tmp7 = tmp6 / tmp1
    tmp8 = triton_helpers.maximum(tmp5, tmp7)
    tmp10 = tmp9 / tmp1
    tmp11 = triton_helpers.maximum(tmp8, tmp10)
    tmp12 = tmp2 > tmp4
    tmp13 = tmp2 == tmp4
    tmp14 = tmp2 != tmp2
    tmp15 = tmp4 != tmp4
    tmp16 = tmp14 > tmp15
    tmp17 = tmp12 | tmp16
    tmp18 = tmp14 & tmp15
    tmp19 = tmp13 | tmp18
    tmp20 = tl.full([1], 0, tl.int64)
    tmp21 = tl.full([1], 1, tl.int64)
    tmp22 = tmp20 < tmp21
    tmp23 = tmp19 & tmp22
    tmp24 = tmp17 | tmp23
    tmp25 = tl.where(tmp24, tmp2, tmp4)
    tmp26 = tl.where(tmp24, tmp20, tmp21)
    tmp27 = tmp25 > tmp7
    tmp28 = tmp25 == tmp7
    tmp29 = tmp25 != tmp25
    tmp30 = tmp7 != tmp7
    tmp31 = tmp29 > tmp30
    tmp32 = tmp27 | tmp31
    tmp33 = tmp29 & tmp30
    tmp34 = tmp28 | tmp33
    tmp35 = tl.full([1], 2, tl.int64)
    tmp36 = tmp26 < tmp35
    tmp37 = tmp34 & tmp36
    tmp38 = tmp32 | tmp37
    tmp39 = tl.where(tmp38, tmp25, tmp7)
    tmp40 = tl.where(tmp38, tmp26, tmp35)
    tmp41 = tmp39 > tmp10
    tmp42 = tmp39 == tmp10
    tmp43 = tmp39 != tmp39
    tmp44 = tmp10 != tmp10
    tmp45 = tmp43 > tmp44
    tmp46 = tmp41 | tmp45
    tmp47 = tmp43 & tmp44
    tmp48 = tmp42 | tmp47
    tmp49 = tl.full([1], 3, tl.int64)
    tmp50 = tmp40 < tmp49
    tmp51 = tmp48 & tmp50
    tmp52 = tmp46 | tmp51
    tmp53 = tl.where(tmp52, tmp39, tmp10)
    tmp54 = tl.where(tmp52, tmp40, tmp49)
    tl.store(out_ptr0 + (x0), tmp11, xmask)
    tl.store(out_ptr1 + (x0), tmp54, xmask)
