
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_layer_norm_8', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 16, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_native_layer_norm_8(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x4 = ((xindex // 4) % 16)
    x3 = xindex // 64
    x5 = (xindex % 16)
    x6 = xindex
    tmp0 = tl.load(in_ptr0 + (4*x4), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (4*x4), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (0))
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK])
    tmp6 = tl.load(in_ptr3 + (4*x5 + 64*x3), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr0 + (1 + 4*x4), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr1 + (1 + 4*x4), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr2 + (1))
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK])
    tmp14 = tl.load(in_ptr3 + (1 + 4*x5 + 64*x3), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr0 + (2 + 4*x4), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr1 + (2 + 4*x4), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr2 + (2))
    tmp20 = tl.broadcast_to(tmp19, [XBLOCK])
    tmp23 = tl.load(in_ptr3 + (2 + 4*x5 + 64*x3), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr0 + (3 + 4*x4), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr1 + (3 + 4*x4), xmask, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr2 + (3))
    tmp29 = tl.broadcast_to(tmp28, [XBLOCK])
    tmp32 = tl.load(in_ptr3 + (3 + 4*x5 + 64*x3), xmask, eviction_policy='evict_last')
    tmp4 = tmp1 + tmp3
    tmp5 = tmp0 + tmp4
    tmp7 = tmp5 + tmp6
    tmp12 = tmp9 + tmp11
    tmp13 = tmp8 + tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tmp7 + tmp15
    tmp21 = tmp18 + tmp20
    tmp22 = tmp17 + tmp21
    tmp24 = tmp22 + tmp23
    tmp25 = tmp16 + tmp24
    tmp30 = tmp27 + tmp29
    tmp31 = tmp26 + tmp30
    tmp33 = tmp31 + tmp32
    tmp34 = tmp25 + tmp33
    tmp35 = 4.0
    tmp36 = tmp34 / tmp35
    tmp37 = tmp7 - tmp36
    tmp38 = tmp37 * tmp37
    tmp39 = tmp15 - tmp36
    tmp40 = tmp39 * tmp39
    tmp41 = tmp38 + tmp40
    tmp42 = tmp24 - tmp36
    tmp43 = tmp42 * tmp42
    tmp44 = tmp41 + tmp43
    tmp45 = tmp33 - tmp36
    tmp46 = tmp45 * tmp45
    tmp47 = tmp44 + tmp46
    tmp48 = tmp47 / tmp35
    tl.store(out_ptr0 + (x6), tmp36, xmask)
    tl.store(out_ptr1 + (x6), tmp48, xmask)
