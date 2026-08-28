# WMT26 Model-Compression Leaderboard — wmt26

Reference-free QE on the blind set. **Δ** = system − `baseline--uncompressed` (same direction).

- **ck_xxl** = cometkiwi-XXL (higher is better) · **mx_xxl** = MetricX-24-XXL (lower is better)
- **size GB** on-disk weights · **comp%** = size vs baseline (constrained only) · **thrpt ch/s** = source chars ÷ wall-time at the largest measured batch · **b1 lat s** = batch-1 single-stream wall-time on ces-deu (speed track) · **peak GB** = peak host RSS
- Tracks: **constrained** = compress gemma-3-12b · **unconstrained** = different base model (shown separately)

## ces-deu

### ces-deu — Constrained (compress gemma-3-12b; Δ/comp% vs baseline--uncompressed)

| # | system | ck_xxl | Δ | mx_xxl | Δ | size GB | comp% | thrpt ch/s | b1 lat s | peak GB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | fbk--gptq-rtn | 0.5697 | +0.0071 | 6.9124 | +0.5993 | 13.7 | 56 | 3997 | 3668.4 | 14.2 |
| 2 | tahomamt--fp8-bok4 | 0.5696 | +0.0070 | 6.3723 | +0.0592 | 11.5 | 47 | 2351 | 123.4 | 21.8 |
| 3 | fbk--sq-gptq | 0.5680 | +0.0054 | 6.9146 | +0.6015 | 13.7 | 56 | 4199 | 3806.7 | 14.3 |
| 4 | pare4bit--int4-gptq | 0.5675 | +0.0049 | 6.4558 | +0.1427 | 7.6 | 31 | 1682 | 5743.7 | 8.8 |
| 5 | alonso--layeraware-native-mlp-q4 | 0.5638 | +0.0012 | 6.2349 | -0.0782 | 11.8 | 48 | 341 | 4131.5 | 5.8 |
| 6 | baseline--bnb-q8 | 0.5638 | +0.0012 | 6.2952 | -0.0179 | 13.3 | 54 | 252 | 11958.3 | 5.9 |
| 7 | arc-ilsp--fp8 | 0.5630 | +0.0004 | 6.3109 | -0.0022 | 14.8 | 61 | 3471 | 659.1 | 6.5 |
| 8 | cometcut--gptq | 0.5629 | +0.0003 | 6.4512 | +0.1381 | 8.5 | 35 | 2942 | 733.8 | 9.7 |
| 9 | tiny-titans--gptq-int4-calibrated | 0.5627 | +0.0001 | 6.3552 | +0.0421 | 7.6 | 31 | 937 | 3634.7 | 8.3 |
| 10 | baseline--uncompressed ⟵ | 0.5626 | +0.0000 | 6.3131 | +0.0000 | 24.4 | 100 | 608 | 3722.8 | 5.8 |
| 11 | tahomamt--fp8 | 0.5624 | -0.0002 | 6.4246 | +0.1115 | 11.5 | 47 | 6311 | 835.3 | 21.8 |
| 12 | arc-ilsp--en-ar-mbr | 0.5612 | -0.0014 | 6.3965 | +0.0834 | 7.1 | 29 | 672 | 932.0 | 8.3 |
| 13 | arc-ilsp--vocaball-int4 | 0.5611 | -0.0015 | 6.4244 | +0.1113 | 7.1 | 29 | 3559 | 615.0 | 8.3 |
| 14 | arc-ilsp--int4 | 0.5610 | -0.0016 | 6.5161 | +0.2030 | 7.6 | 31 | 2096 | 642.1 | 6.5 |
| 15 | tahomamt--int4-bok4 | 0.5608 | -0.0018 | 6.4591 | +0.1460 | 6.5 | 27 | 2119 | 141.0 | 12.4 |
| 16 | cometcut--comet-mixed-precision | 0.5598 | -0.0028 | 6.3832 | +0.0701 | 9.2 | 37 | 3115 | 85.5 | 6.4 |
| 17 | pare4bit--pruned4-healed-int4 | 0.5569 | -0.0057 | 7.4080 | +1.0949 | 7.1 | 29 | 2003 | 5218.2 | 8.4 |
| 18 | baseline--bnb-q4 | 0.5547 | -0.0079 | 6.4872 | +0.1741 | 8.4 | 34 | 811 | 3611.4 | 5.7 |
| 19 | cometcut--awq | 0.5533 | -0.0093 | 6.4300 | +0.1169 | 8.5 | 35 | 3291 | 734.3 | 9.7 |
| 20 | tahomamt--int4 | 0.5492 | -0.0134 | 6.5208 | +0.2077 | 6.5 | 26 | 5538 | 772.0 | 13.0 |
| 21 | vicomtech--ratio_0.25_ffn_awq_4bits | 0.5370 | -0.0256 | 10.3737 | +4.0606 | 6.1 | 25 | 5791 | 385.1 | 7.3 |
| 22 | vicomtech--ratio_0.50_ffn_sft_checkpoint-4802_awq_4bits | 0.4433 | -0.1193 | 14.3829 | +8.0698 | 4.6 | 19 | 6088 | 276.0 | 5.9 |
| 23 | vicomtech--ratio_0.50_ffn_sft_checkpoint-4802_awq_4bits_ces-deu | 0.4428 | -0.1198 | 14.2350 | +7.9219 | 3.9 | 16 | 6477 | 263.7 | 5.0 |
| 24 | vicomtech--ratio_0.50_ffn_sft_checkpoint-4802_awq_4bits_eng-zho | 0.4428 | -0.1198 | 14.5909 | +8.2778 | 4.0 | 16 | 6090 | 262.6 | 5.2 |
| 25 | vicomtech--ratio_0.50_ffn_sft_checkpoint-4802_awq_4bits_eng-ara | 0.4426 | -0.1200 | 14.2350 | +7.9219 | 3.9 | 16 | 6761 | 267.0 | 5.1 |
| 26 | tiny-titans--svd-qkv512-ff1620 | 0.3153 | -0.2473 | 19.7835 | +13.4704 | 12.6 | 52 | 17 | 833.9 | 12.7 |
| 27 | tmu-onono--diba-triton_direct | 0.2787 | -0.2839 | 18.8473 | +12.5342 | 2.5 | 10 | 445 | 4097.4 | 3.4 |
| 28 | tmu-onono--diba-cached_unpacked | 0.2740 | -0.2886 | 18.8046 | +12.4915 | 2.5 | 10 | 3160 | 1725.6 | 5.7 |
| 29 | vicomtech--ratio_0.50_ffn_awq_4bits | 0.2647 | -0.2979 | 16.7606 | +10.4475 | 4.6 | 19 | 6106 | 337.9 | 5.9 |
| 30 | tiny-titans--svd-qkv512-ff1620-q4-g32-full | 0.2431 | -0.3195 | 21.6792 | +15.3661 | 5.4 | 22 | 877 | 525.4 | 5.9 |
| 31 | slicers--navadeep-gemma3-12b | 0.1054 | -0.4572 | 22.3829 | +16.0698 | 16.6 | 68 | 305 | 2852.0 | 16.5 |

### ces-deu — Unconstrained (different base model; not a compression ratio)

| # | system | ck_xxl | Δ | mx_xxl | Δ | size GB | thrpt ch/s | b1 lat s | peak GB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | tildeopen--15B-GPTQ-nvfp4a16 | 0.5926 | +0.0300 | 6.2183 | -0.0948 | 10.1 | 1407 | 2274.6 | 10.7 |
| 2 | tildeopen--15B-RTN-fp8-dyn | 0.5885 | +0.0259 | 6.2682 | -0.0449 | 16.3 | 1448 | 2527.1 | 16.4 |
| 3 | pare4bit--unconstrained-gemma4 | 0.5865 | +0.0239 | 6.0834 | -0.2297 | 10.3 | 996 | 3150.7 | 9.8 |
| 4 | tildeopen--8B-distill | 0.5811 | +0.0185 | 6.3056 | -0.0075 | 16.3 | 3552 | 2240.3 | 16.5 |
| 5 | tildeopen--8B-distill-GPTQ-nvfp4a16 | 0.5796 | +0.0170 | 6.3908 | +0.0777 | 5.7 | 2228 | 2209.4 | 6.6 |
| 6 | tildeopen--8B-distill-RTN-fp8-dyn | 0.5794 | +0.0168 | 6.2730 | -0.0401 | 8.9 | 2891 | 2412.6 | 9.6 |

## eng-zho_Hans

### eng-zho_Hans — Constrained (compress gemma-3-12b; Δ/comp% vs baseline--uncompressed)

| # | system | ck_xxl | Δ | mx_xxl | Δ | size GB | comp% | thrpt ch/s | b1 lat s | peak GB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | tahomamt--fp8-bok4 | 0.7749 | +0.0054 | 2.6568 | -0.1160 | 11.5 | 47 | 2558 | — | 21.8 |
| 2 | tahomamt--int4-bok4 | 0.7706 | +0.0011 | 2.6904 | -0.0824 | 6.5 | 27 | 2473 | — | 12.4 |
| 3 | cometcut--comet-mixed-precision | 0.7698 | +0.0003 | 2.7512 | -0.0216 | 9.2 | 37 | 3097 | — | 7.5 |
| 4 | baseline--uncompressed ⟵ | 0.7695 | +0.0000 | 2.7728 | +0.0000 | 24.4 | 100 | 881 | — | 5.8 |
| 5 | arc-ilsp--fp8 | 0.7691 | -0.0004 | 2.7486 | -0.0242 | 14.8 | 61 | 3901 | — | 7.5 |
| 6 | baseline--bnb-q8 | 0.7684 | -0.0011 | 2.7986 | +0.0258 | 13.3 | 54 | 508 | — | 5.9 |
| 7 | tahomamt--fp8 | 0.7674 | -0.0021 | 2.7824 | +0.0096 | 11.5 | 47 | 6052 | — | 21.8 |
| 8 | alonso--layeraware-native-mlp-q4 | 0.7669 | -0.0026 | 2.7836 | +0.0108 | 11.8 | 48 | 867 | — | 5.7 |
| 9 | baseline--bnb-q4 | 0.7665 | -0.0030 | 2.8147 | +0.0419 | 8.4 | 34 | 868 | — | 7.5 |
| 10 | cometcut--gptq | 0.7655 | -0.0040 | 2.7600 | -0.0128 | 8.5 | 35 | 3166 | — | 9.7 |
| 11 | arc-ilsp--int4 | 0.7646 | -0.0049 | 2.8467 | +0.0739 | 7.6 | 31 | 3537 | — | 6.5 |
| 12 | pare4bit--int4-gptq | 0.7644 | -0.0051 | 2.7772 | +0.0044 | 7.6 | 31 | 1931 | — | 8.8 |
| 13 | cometcut--awq | 0.7602 | -0.0093 | 2.8177 | +0.0449 | 8.5 | 35 | 3137 | — | 9.7 |
| 14 | pare4bit--pruned4-healed-int4 | 0.7583 | -0.0112 | 2.9739 | +0.2011 | 7.1 | 29 | 1856 | — | 8.4 |
| 15 | tahomamt--int4 | 0.7574 | -0.0121 | 2.8207 | +0.0479 | 6.5 | 26 | 4879 | — | 13.0 |
| 16 | vicomtech--ratio_0.25_ffn_awq_4bits | 0.7248 | -0.0447 | 3.6745 | +0.9017 | 6.1 | 25 | 5127 | — | 7.5 |
| 17 | vicomtech--ratio_0.50_ffn_sft_checkpoint-4802_awq_4bits_eng-zho | 0.5805 | -0.1890 | 5.7679 | +2.9951 | 4.0 | 16 | 4133 | — | 5.2 |
| 18 | vicomtech--ratio_0.50_ffn_sft_checkpoint-4802_awq_4bits | 0.5804 | -0.1891 | 5.7004 | +2.9276 | 4.6 | 19 | 5598 | — | 7.2 |
| 19 | slicers--navadeep-gemma3-12b | 0.5688 | -0.2007 | 5.6239 | +2.8511 | 16.6 | 68 | 825 | — | 16.6 |
| 20 | vicomtech--ratio_0.50_ffn_awq_4bits | 0.5557 | -0.2138 | 5.7350 | +2.9622 | 4.6 | 19 | 5244 | — | 7.5 |

### eng-zho_Hans — Unconstrained (different base model; not a compression ratio)

| # | system | ck_xxl | Δ | mx_xxl | Δ | size GB | thrpt ch/s | b1 lat s | peak GB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | pare4bit--unconstrained-gemma4 | 0.7908 | +0.0213 | 2.6241 | -0.1487 | 10.3 | 951 | — | 9.8 |
| 2 | ests--gptoss-zho-k26 | 0.7244 | -0.0451 | 3.3541 | +0.5813 | 5.5 | 319 | — | 5.9 |
| 3 | ests--gptoss-zho-k27 | 0.7140 | -0.0555 | 3.4451 | +0.6723 | 5.2 | 310 | — | 6.0 |
| 4 | ests--gptoss-zho-k28 | 0.6890 | -0.0805 | 3.7355 | +0.9627 | 4.9 | 249 | — | 5.8 |
| 5 | dragon1006--aya-expanse-8b-int8 | 0.6602 | -0.1093 | 4.5167 | +1.7439 | 8.2 | 590 | — | 8.7 |

## eng-ara_EG

### eng-ara_EG — Constrained (compress gemma-3-12b; Δ/comp% vs baseline--uncompressed)

| # | system | ck_xxl | Δ | mx_xxl | Δ | size GB | comp% | thrpt ch/s | b1 lat s | peak GB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | vicomtech--ratio_0.25_ffn_awq_4bits | 0.7069 | +0.0918 | 5.6286 | +0.2647 | 6.1 | 25 | 4125 | — | 7.5 |
| 2 | tahomamt--int4-bok4 | 0.6431 | +0.0280 | 5.0776 | -0.2863 | 6.5 | 27 | 2037 | — | 12.4 |
| 3 | tahomamt--fp8-bok4 | 0.6312 | +0.0161 | 5.2881 | -0.0758 | 11.5 | 47 | 2317 | — | 21.8 |
| 4 | tahomamt--int4 | 0.6211 | +0.0060 | 5.4629 | +0.0990 | 6.5 | 26 | 4253 | — | 13.0 |
| 5 | pare4bit--pruned4-healed-int4 | 0.6199 | +0.0048 | 5.7265 | +0.3626 | 7.1 | 29 | 1495 | — | 8.4 |
| 6 | baseline--uncompressed ⟵ | 0.6151 | +0.0000 | 5.3639 | +0.0000 | 24.4 | 100 | 717 | — | 5.8 |
| 7 | arc-ilsp--en-ar-mbr | 0.6142 | -0.0009 | 5.5382 | +0.1743 | 7.1 | 29 | 767 | — | 8.3 |
| 8 | baseline--bnb-q8 | 0.6105 | -0.0046 | 5.4909 | +0.1270 | 13.3 | 54 | 418 | — | 5.9 |
| 9 | cometcut--awq | 0.6099 | -0.0052 | 5.6717 | +0.3078 | 8.5 | 35 | 2665 | — | 9.7 |
| 10 | tahomamt--fp8 | 0.6099 | -0.0052 | 5.4918 | +0.1279 | 11.5 | 47 | 4410 | — | 21.8 |
| 11 | arc-ilsp--fp8 | 0.6084 | -0.0067 | 5.5100 | +0.1461 | 14.8 | 61 | 2871 | — | 7.5 |
| 12 | baseline--bnb-q4 | 0.6066 | -0.0085 | 5.6233 | +0.2594 | 8.4 | 34 | 722 | — | 7.5 |
| 13 | cometcut--gptq | 0.6058 | -0.0093 | 5.6166 | +0.2527 | 8.5 | 35 | 2546 | — | 9.7 |
| 14 | cometcut--comet-mixed-precision | 0.6019 | -0.0132 | 5.4627 | +0.0988 | 9.2 | 37 | 2486 | — | 7.5 |
| 15 | pare4bit--int4-gptq | 0.6014 | -0.0137 | 5.7105 | +0.3466 | 7.6 | 31 | 1401 | — | 8.8 |
| 16 | arc-ilsp--vocaball-int4 | 0.6010 | -0.0141 | 5.6693 | +0.3054 | 7.1 | 29 | 3526 | — | 8.3 |
| 17 | vicomtech--ratio_0.50_ffn_sft_checkpoint-4802_awq_4bits | 0.5830 | -0.0321 | 7.6735 | +2.3096 | 4.6 | 19 | 4740 | — | 7.2 |
| 18 | vicomtech--ratio_0.50_ffn_sft_checkpoint-4802_awq_4bits_eng-ara | 0.5826 | -0.0325 | 7.6860 | +2.3221 | 3.9 | 16 | 3330 | — | 5.1 |
| 19 | vicomtech--ratio_0.50_ffn_awq_4bits | 0.4885 | -0.1266 | 10.0958 | +4.7319 | 4.6 | 19 | 4601 | — | 7.5 |

### eng-ara_EG — Unconstrained (different base model; not a compression ratio)

| # | system | ck_xxl | Δ | mx_xxl | Δ | size GB | thrpt ch/s | b1 lat s | peak GB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | pare4bit--unconstrained-gemma4 | 0.6822 | +0.0671 | 4.4869 | -0.8770 | 10.3 | 1262 | — | 9.8 |
| 2 | ests--gptoss-arz-k22 | 0.6202 | +0.0051 | 5.8402 | +0.4763 | 6.8 | 503 | — | 6.0 |
| 3 | ests--gptoss-arz-k24 | 0.6081 | -0.0070 | 6.0546 | +0.6907 | 6.2 | 100 | — | 5.9 |
| 4 | ests--gptoss-arz-k26 | 0.5678 | -0.0473 | 7.0141 | +1.6502 | 5.5 | 94 | — | 5.9 |

