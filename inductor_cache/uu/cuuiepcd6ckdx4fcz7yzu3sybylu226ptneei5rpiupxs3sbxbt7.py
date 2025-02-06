
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': (4,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_div_mean_rsub_sub_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 12, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_div_mean_rsub_sub_1(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp2 = tl.load(in_ptr1 + (0))
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK])
    tmp5 = tl.load(in_ptr2 + (0))
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK])
    tmp10 = tl.load(in_ptr0 + (1))
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK])
    tmp12 = tl.load(in_ptr1 + (1))
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK])
    tmp15 = tl.load(in_ptr2 + (1))
    tmp16 = tl.broadcast_to(tmp15, [XBLOCK])
    tmp20 = tl.load(in_ptr0 + (2))
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK])
    tmp22 = tl.load(in_ptr1 + (2))
    tmp23 = tl.broadcast_to(tmp22, [XBLOCK])
    tmp25 = tl.load(in_ptr2 + (2))
    tmp26 = tl.broadcast_to(tmp25, [XBLOCK])
    tmp30 = tl.load(in_ptr0 + (3))
    tmp31 = tl.broadcast_to(tmp30, [XBLOCK])
    tmp32 = tl.load(in_ptr1 + (3))
    tmp33 = tl.broadcast_to(tmp32, [XBLOCK])
    tmp35 = tl.load(in_ptr2 + (3))
    tmp36 = tl.broadcast_to(tmp35, [XBLOCK])
    tmp4 = tmp1 / tmp3
    tmp7 = tmp4 - tmp6
    tmp8 = 1.0
    tmp9 = tmp8 - tmp7
    tmp14 = tmp11 / tmp13
    tmp17 = tmp14 - tmp16
    tmp18 = tmp8 - tmp17
    tmp19 = tmp9 + tmp18
    tmp24 = tmp21 / tmp23
    tmp27 = tmp24 - tmp26
    tmp28 = tmp8 - tmp27
    tmp29 = tmp19 + tmp28
    tmp34 = tmp31 / tmp33
    tmp37 = tmp34 - tmp36
    tmp38 = tmp8 - tmp37
    tmp39 = tmp29 + tmp38
    tmp40 = 4.0
    tmp41 = tmp39 / tmp40
    tl.store(out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp41, None)
