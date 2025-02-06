
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_copy_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_copy_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 24
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 6)
    x1 = xindex // 6
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 5, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = (-4) + x0
    tmp4 = tl.full([1], 1, tl.int64)
    tmp5 = tmp3 < tmp4
    tmp6 = tmp5 & tmp2
    tmp7 = x0
    tmp8 = tl.full([1], 1, tl.int64)
    tmp9 = tmp7 >= tmp8
    tmp10 = tl.full([1], 5, tl.int64)
    tmp11 = tmp7 < tmp10
    tmp12 = tmp9 & tmp11
    tmp13 = tmp12 & tmp6
    tmp14 = tl.load(in_ptr0 + ((-1) + x0 + 4*x1), tmp13 & xmask, other=0.0)
    tmp15 = 16.0
    tmp16 = tmp14 / tmp15
    tmp17 = tl.full(tmp16.shape, 0.0, tmp16.dtype)
    tmp18 = tl.where(tmp13, tmp16, tmp17)
    tmp19 = float("nan")
    tmp20 = tl.where(tmp12, tmp18, tmp19)
    tmp21 = tl.full(tmp20.shape, 0.0, tmp20.dtype)
    tmp22 = tl.where(tmp6, tmp20, tmp21)
    tmp23 = tmp3 >= tmp4
    tmp24 = tl.full([1], 5, tl.int64)
    tmp25 = tmp3 < tmp24
    tmp26 = tmp23 & tmp25
    tmp27 = tmp26 & tmp2
    tmp28 = tl.load(in_ptr0 + ((-5) + x0 + 4*x1), tmp27 & xmask, other=0.0)
    tmp29 = 16.0
    tmp30 = tmp28 / tmp29
    tmp31 = tl.full(tmp30.shape, 0.0, tmp30.dtype)
    tmp32 = tl.where(tmp27, tmp30, tmp31)
    tmp33 = float("nan")
    tmp34 = tl.where(tmp26, tmp32, tmp33)
    tmp35 = tl.where(tmp5, tmp22, tmp34)
    tmp36 = tl.full(tmp35.shape, 0.0, tmp35.dtype)
    tmp37 = tl.where(tmp2, tmp35, tmp36)
    tmp38 = tl.full([1], 1, tl.int64)
    tmp39 = tmp0 < tmp38
    tmp40 = 4 + x0
    tmp41 = tl.full([1], 1, tl.int64)
    tmp42 = tmp40 >= tmp41
    tmp43 = tl.full([1], 5, tl.int64)
    tmp44 = tmp40 < tmp43
    tmp45 = tmp42 & tmp44
    tmp46 = tmp45 & tmp39
    tmp47 = tl.load(in_ptr0 + (3 + x0 + 4*x1), tmp46 & xmask, other=0.0)
    tmp48 = 16.0
    tmp49 = tmp47 / tmp48
    tmp50 = tl.full(tmp49.shape, 0.0, tmp49.dtype)
    tmp51 = tl.where(tmp46, tmp49, tmp50)
    tmp52 = float("nan")
    tmp53 = tl.where(tmp45, tmp51, tmp52)
    tmp54 = tl.full(tmp53.shape, 0.0, tmp53.dtype)
    tmp55 = tl.where(tmp39, tmp53, tmp54)
    tmp56 = tmp0 >= tmp38
    tmp57 = tmp0 < tmp1
    tmp58 = tmp56 & tmp57
    tmp59 = tl.load(in_ptr0 + ((-1) + x0 + 4*x1), tmp58 & xmask, other=0.0)
    tmp60 = 16.0
    tmp61 = tmp59 / tmp60
    tmp62 = tl.full(tmp61.shape, 0.0, tmp61.dtype)
    tmp63 = tl.where(tmp58, tmp61, tmp62)
    tmp64 = float("nan")
    tmp65 = tl.where(tmp58, tmp63, tmp64)
    tmp66 = tl.where(tmp39, tmp55, tmp65)
    tmp67 = tl.where(tmp2, tmp37, tmp66)
    tl.store(out_ptr0 + (x2), tmp67, xmask)
