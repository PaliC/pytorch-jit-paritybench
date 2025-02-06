
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 8), 'tt.equal_to': (7,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_abs_add_div_mean_mse_loss_mul_sub_36', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 7, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_abs_add_div_mean_mse_loss_mul_sub_36(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, rnumel):
    xnumel = 1
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    r0 = rindex
    tmp0 = tl.load(in_ptr0 + (r0), None)
    tmp4 = tl.load(in_out_ptr0 + (0))
    tmp5 = tl.broadcast_to(tmp4, [1])
    tmp8 = tl.load(in_ptr1 + (0))
    tmp9 = tl.broadcast_to(tmp8, [1])
    tmp13 = tl.load(in_ptr2 + (0))
    tmp14 = tl.broadcast_to(tmp13, [1])
    tmp21 = tl.load(in_ptr3 + (0))
    tmp22 = tl.broadcast_to(tmp21, [1])
    tmp28 = tl.load(in_ptr4 + (0))
    tmp29 = tl.broadcast_to(tmp28, [1])
    tmp35 = tl.load(in_ptr5 + (0))
    tmp36 = tl.broadcast_to(tmp35, [1])
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = triton_helpers.promote_to_tensor(tl.sum(tmp1, 0))
    tmp6 = 49152.0
    tmp7 = tmp5 / tmp6
    tmp10 = 131072.0
    tmp11 = tmp9 / tmp10
    tmp12 = tmp7 + tmp11
    tmp15 = 65536.0
    tmp16 = tmp14 / tmp15
    tmp17 = 0.03125
    tmp18 = tmp16 * tmp17
    tmp19 = 0.0
    tmp20 = tmp18 + tmp19
    tmp23 = 262144.0
    tmp24 = tmp22 / tmp23
    tmp25 = 0.0625
    tmp26 = tmp24 * tmp25
    tmp27 = tmp20 + tmp26
    tmp30 = 1048576.0
    tmp31 = tmp29 / tmp30
    tmp32 = 0.125
    tmp33 = tmp31 * tmp32
    tmp34 = tmp27 + tmp33
    tmp37 = 4194304.0
    tmp38 = tmp36 / tmp37
    tmp39 = 1.0
    tmp40 = tmp38 * tmp39
    tmp41 = tmp34 + tmp40
    tmp42 = tmp3 / tmp37
    tmp43 = tmp42 * tmp39
    tmp44 = tmp41 + tmp43
    tmp45 = 100.0
    tmp46 = tmp44 * tmp45
    tmp47 = tmp12 + tmp46
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([1], 0, tl.int32)), tmp47, None)
