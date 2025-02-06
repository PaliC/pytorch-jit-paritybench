
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mv_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 13, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mv_3(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (4*(((x0 // 4) % 4)) + 64*((x0 % 4)) + (x0 // 16)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x0 // 16), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (16*(((x0 // 4) % 4)) + ((x0 % 4))), xmask)
    tmp6 = tl.load(in_ptr3 + (0))
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK])
    tmp9 = tl.load(in_ptr0 + (16 + 4*(((x0 // 4) % 4)) + 64*((x0 % 4)) + (x0 // 16)), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr2 + (4 + 16*(((x0 // 4) % 4)) + ((x0 % 4))), xmask)
    tmp14 = tl.load(in_ptr3 + (1))
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK])
    tmp18 = tl.load(in_ptr0 + (32 + 4*(((x0 // 4) % 4)) + 64*((x0 % 4)) + (x0 // 16)), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr2 + (8 + 16*(((x0 // 4) % 4)) + ((x0 % 4))), xmask)
    tmp23 = tl.load(in_ptr3 + (2))
    tmp24 = tl.broadcast_to(tmp23, [XBLOCK])
    tmp27 = tl.load(in_ptr0 + (48 + 4*(((x0 // 4) % 4)) + 64*((x0 % 4)) + (x0 // 16)), xmask, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr2 + (12 + 16*(((x0 // 4) % 4)) + ((x0 % 4))), xmask)
    tmp32 = tl.load(in_ptr3 + (3))
    tmp33 = tl.broadcast_to(tmp32, [XBLOCK])
    tmp3 = libdevice.tanh(tmp2)
    tmp4 = tmp1 * tmp3
    tmp5 = tmp0 + tmp4
    tmp8 = tmp5 * tmp7
    tmp11 = libdevice.tanh(tmp10)
    tmp12 = tmp1 * tmp11
    tmp13 = tmp9 + tmp12
    tmp16 = tmp13 * tmp15
    tmp17 = tmp8 + tmp16
    tmp20 = libdevice.tanh(tmp19)
    tmp21 = tmp1 * tmp20
    tmp22 = tmp18 + tmp21
    tmp25 = tmp22 * tmp24
    tmp26 = tmp17 + tmp25
    tmp29 = libdevice.tanh(tmp28)
    tmp30 = tmp1 * tmp29
    tmp31 = tmp27 + tmp30
    tmp34 = tmp31 * tmp33
    tmp35 = tmp26 + tmp34
    tl.store(out_ptr0 + (x0), tmp35, xmask)
