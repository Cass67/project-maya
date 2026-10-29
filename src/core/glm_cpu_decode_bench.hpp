#pragma once
// Full-model CPU-cache benchmark: one prompt snapshot, a native greedy reference, 64-row exact parity,
// then off/on/on/off MTP trials. Requires background promotions off for stable expert placement.
inline bool glm_cpu_decode_bench(strata::core::Glm5Model& model, const std::vector<int32_t>& ids,
                                  int prefix_count, int count, const char* thread_list, std::string& err) {
    if (prefix_count < 2 || ids.empty() || count < 64) { err="decode benchmark needs prompt IDs and at least 64 outputs"; return false; }
    std::vector<int32_t> prompt((size_t)prefix_count-1);
    for(int i=0;i<prefix_count-1;++i)prompt[(size_t)i]=ids[(size_t)i%ids.size()];
    const int32_t last=ids[((size_t)prefix_count-1)%ids.size()];
    model.set_cpu_plane_cache(false); model.set_pcie_share(0.0); model.set_follow({}); model.reset();
    auto begin=std::chrono::steady_clock::now();
    if(!model.prefill(prompt,err,last) || !model.snapshot_save()) { if(err.empty())err="prompt snapshot failed";return false; }
    std::printf("CPU_AB prompt=%d snapshot_pos=%lld prefill_s=%.3f\n",prefix_count,(long long)model.snapshot_pos(),
        std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count());std::fflush(stdout);
    const size_t V=(size_t)model.geometry().n_vocab;
    std::vector<float> lg, reference_logits;
    reference_logits.reserve(64*V);
    std::vector<int32_t> reference;
    strata::kernels::SamplerParams sp;sp.temperature=0;sp.greedy=true;
    if(!model.forward({last},lg,err))return false;
    // Permit PROMOTE_FILL during this warmup, with PROMOTE=0: empty post-prefill slots fill, then no hot/cold swaps.
    const int warm_tokens=std::min(count,128);
    for(int i=0;i<warm_tokens;++i){
        int tok=model.sample_token(sp,err);if(tok<0)return false;
        if(i+1<warm_tokens && !model.forward({tok},lg,err))return false;
    }
    model.set_follow({});
    if(!model.snapshot_restore() || !model.forward({last},lg,err)){if(err.empty())err="warmup snapshot restore failed";return false;}
    std::printf("CPU_AB warm tokens=%d complete\n",warm_tokens);std::fflush(stdout);
    const auto original_prefix=lg;
    begin=std::chrono::steady_clock::now();
    for(int i=0;i<count;++i){
        if(i<64)reference_logits.insert(reference_logits.end(),lg.begin(),lg.end());
        int tok=model.sample_token(sp,err);if(tok<0)return false;reference.push_back(tok);
        if(i+1<count && !model.forward({tok},lg,err))return false;
    }
    std::printf("CPU_AB native_reference tokens=%d tok_s=%.3f\n",count,
        count/std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count());std::fflush(stdout);
    // Independent serial execution on the plane backend, all vocabulary logits at 64 teacher-forced positions.
    model.set_cpu_plane_cache(true);model.set_follow({});
    if(!model.snapshot_restore() || !model.forward({last},lg,err)){if(err.empty())err="parity snapshot restore failed";return false;}
    size_t differing=0;float worst=0;
    for(int i=0;i<64;++i){
        for(size_t v=0;v<V;++v){float a=reference_logits[(size_t)i*V+v],b=lg[v];
            if(!std::isfinite(a)||!std::isfinite(b)){err="nonfinite parity logits";return false;}
            differing+=std::memcmp(&a,&b,sizeof(float))!=0;worst=std::max(worst,std::fabs(a-b));}
        if(i+1<64 && !model.forward({reference[(size_t)i]},lg,err))return false;
    }
    std::printf("CPU_AB parity rows=64 vocabulary=%zu differing_values=%zu max_abs=%.9g\n",V,differing,worst);std::fflush(stdout);
    if(differing){err="CPU planes differ from native full-model logits";return false;}
    for(const char*q=thread_list;*q;){
        const int threads=std::atoi(q);
        if(threads>0)for(int trial=0;trial<4;++trial){
            const bool on=trial==1||trial==2;
            model.set_cpu_plane_cache(on);model.set_cpu_lane_threads(threads);model.set_pcie_share(0.0);model.set_follow({});
            if(!model.snapshot_restore() || !model.forward({last},lg,err)){if(err.empty())err="A/B snapshot restore failed";return false;}
            // Different worker counts must not change this identical prompt's next-token logits.
            if(lg.size()!=original_prefix.size() || std::memcmp(lg.data(),original_prefix.data(),lg.size()*sizeof(float))){
                err="restored prefix logits changed";return false;
            }
            model.set_follow(reference);
            std::vector<int> generated;
            const auto before=model.fast_stats();const uint64_t offers=model.spec_steps_,accepts=model.spec_hits_;
            std::fprintf(stderr,"CPU_AB BEGIN planes=%d threads=%d trial=%d tokens=%d\n",on,threads,trial,count);
            begin=std::chrono::steady_clock::now();int64_t produced=0;
            if(!model.decode_spec(sp,count,[&](int tok){generated.push_back(tok);return true;},produced,err))return false;
            double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
            const auto after=model.fast_stats();double n=(double)generated.size();
            bool follows=generated.size()==reference.size() && std::equal(generated.begin(),generated.end(),reference.begin());
            std::printf("CPU_AB planes=%d threads=%d trial=%d tokens=%zu tok_s=%.3f cpu_experts_tok=%.3f cpu_ms_tok=%.3f misses_tok=%.3f drafts=%llu accepted=%llu follows=%d target_pick_differences=%zu\n",
                on,threads,trial,generated.size(),n*1000/ms,(after.cpu_experts-before.cpu_experts)/n,
                (after.cpu_ms-before.cpu_ms)/n,(after.misses-before.misses)/n,
                (unsigned long long)(model.spec_steps_-offers),(unsigned long long)(model.spec_hits_-accepts),follows,model.follow_differ());std::fflush(stdout);
            if(!follows){err="A/B token sequence mismatch";return false;}
        }
        while(*q && *q!=',')++q;if(*q==',')++q;
    }
    return true;
}
