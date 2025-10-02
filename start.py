from htnr_brain import install, XOpts

x = XOpts(
    # SDR
    sdr_filter_enable=True, sdr_bits=256, sdr_multi_hash=2, sdr_prefilter=4096,
    sdr_pack_uint64=True, sdr_union_cap=16384, sdr_rebuild_every=200,
    # Gating / inhibition / plasticity
    predictive_merge_enable=True, predictive_beta=0.35,
    inhibit_enable=True, inhibit_topN=32, inhibit_gamma=0.22, inhibit_iters=1,
    neuromod_enable=True, neuromod_depth=0.5, neuromod_scale=2.0, neuromod_alpha_floor=0.60,
    # Maintenance
    maintenance_brain_enable=True, maint_tile_rows=1536, maint_sample_pairs=50000,
    # Streaming / pinned safeguards
    stream_cover_ratio=0.5, stream_chunk_rows=0, stream_use_fp16=False,
    pinned_bytes_max=256*1024*1024,
)
mem = HTNRMemoryV55X(embedder, hyper=Hyper(use_torch=True, batch_freeze_enable=True))
install(mem, x)
