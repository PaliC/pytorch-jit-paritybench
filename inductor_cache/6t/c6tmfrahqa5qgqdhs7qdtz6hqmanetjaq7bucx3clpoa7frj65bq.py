
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_20', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_20(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 16) % 132)
    x0 = (xindex % 16)
    x2 = xindex // 2112
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 96, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = x1
    tmp6 = tl.full([1], 0, tl.int64)
    tmp7 = tmp5 >= tmp6
    tmp8 = tl.full([1], 48, tl.int64)
    tmp9 = tmp5 < tmp8
    tmp10 = tmp9 & tmp4
    tmp11 = tl.load(in_ptr0 + (x0 + 16*(x1) + 768*x2), tmp10 & xmask, other=0.0)
    tmp12 = 0.0
    tmp13 = tmp11 > tmp12
    tmp14 = 0.2
    tmp15 = tmp11 * tmp14
    tmp16 = tl.where(tmp13, tmp11, tmp15)
    tmp17 = tl.full(tmp16.shape, 0.0, tmp16.dtype)
    tmp18 = tl.where(tmp10, tmp16, tmp17)
    tmp19 = tmp5 >= tmp8
    tmp20 = tl.full([1], 96, tl.int64)
    tmp21 = tmp5 < tmp20
    tmp22 = tmp19 & tmp4
    tmp23 = tl.load(in_ptr1 + (x0 + 16*((-48) + (x1)) + 768*x2), tmp22 & xmask, other=0.0)
    tmp24 = 0.0
    tmp25 = tmp23 > tmp24
    tmp26 = 0.2
    tmp27 = tmp23 * tmp26
    tmp28 = tl.where(tmp25, tmp23, tmp27)
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp22, tmp28, tmp29)
    tmp31 = tl.where(tmp9, tmp18, tmp30)
    tmp32 = tl.full(tmp31.shape, 0.0, tmp31.dtype)
    tmp33 = tl.where(tmp4, tmp31, tmp32)
    tmp34 = tmp0 >= tmp3
    tmp35 = tl.full([1], 132, tl.int64)
    tmp36 = tmp0 < tmp35
    tmp37 = (-96) + x1
    tmp38 = tl.full([1], 0, tl.int64)
    tmp39 = tmp37 >= tmp38
    tmp40 = tl.full([1], 18, tl.int64)
    tmp41 = tmp37 < tmp40
    tmp42 = tmp41 & tmp34
    tmp43 = tl.load(in_ptr2 + (x0 + 16*((-96) + x1) + 288*x2), tmp42 & xmask, other=0.0)
    tmp44 = 0.0
    tmp45 = tmp43 > tmp44
    tmp46 = 0.2
    tmp47 = tmp43 * tmp46
    tmp48 = tl.where(tmp45, tmp43, tmp47)
    tmp49 = tl.full(tmp48.shape, 0.0, tmp48.dtype)
    tmp50 = tl.where(tmp42, tmp48, tmp49)
    tmp51 = tmp37 >= tmp40
    tmp52 = tl.full([1], 36, tl.int64)
    tmp53 = tmp37 < tmp52
    tmp54 = tmp51 & tmp34
    tmp55 = tl.load(in_ptr3 + (x0 + 16*((-18) + ((-96) + x1)) + 288*x2), tmp54 & xmask, other=0.0)
    tmp56 = 0.0
    tmp57 = tmp55 > tmp56
    tmp58 = 0.2
    tmp59 = tmp55 * tmp58
    tmp60 = tl.where(tmp57, tmp55, tmp59)
    tmp61 = tl.full(tmp60.shape, 0.0, tmp60.dtype)
    tmp62 = tl.where(tmp54, tmp60, tmp61)
    tmp63 = tl.where(tmp41, tmp50, tmp62)
    tmp64 = tl.full(tmp63.shape, 0.0, tmp63.dtype)
    tmp65 = tl.where(tmp34, tmp63, tmp64)
    tmp66 = tl.where(tmp4, tmp33, tmp65)
    tl.store(out_ptr0 + (x3), tmp66, xmask)
