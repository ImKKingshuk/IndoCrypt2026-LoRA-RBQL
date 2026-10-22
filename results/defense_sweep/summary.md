# Utility-vs-Leakage Defense Sweep

This table reports parameter leakage and prediction utility under increasingly restricted APIs. Low parameter error means the adapter remains recoverable; high functional agreement or API label utility means the API remains useful for prediction.

| setting | API | restriction | cases | delta relerr mean | functional agreement mean | API label utility mean | observable fraction mean | failure rate mean |
|---|---|---|---:|---:|---:|---:|---:|---:|
| bert-base-uncased:20newsgroups:endpoint_head | endpoint_logits | exact_scores | 10 | 6.52e-10 | 1.0000 | 0.8493 | 1.0000 | 0.00 |
| bert-base-uncased:20newsgroups:endpoint_head | endpoint_logits_noisy | sigma=0.01 | 10 | 3.41e+03 | 0.9937 | 0.8493 | 1.0000 | 0.00 |
| bert-base-uncased:20newsgroups:endpoint_head | endpoint_logits_rounded | decimals=3 | 10 | 1.23e+02 | 0.9987 | 0.8493 | 1.0000 | 0.00 |
| bert-base-uncased:glue_sst2:hidden_lora | endpoint_logits_hidden_lora | final_logits_only | 3 | n/a | 1.0000 | 0.8793 | 0.0026 | 1.00 |
| bert-base-uncased:glue_sst2:module_lora | module_outputs_hidden_lora | module_outputs | 3 | 8.46e-15 | 1.0000 | 0.8793 | 1.0000 | 0.00 |
| digits_public_feature_head | hard_label | top1_index,no_scores | 10 | n/a | n/a | 1.0000 | 0.0000 | 1.00 |
| digits_public_feature_head | raw_logits | exact_scores | 10 | 9.25e-15 | 1.0000 | 1.0000 | 1.0000 | 0.00 |
| digits_public_feature_head | scored_topk_dataset_queries | k=2,exact_scores | 10 | 2.45e-15 | 1.0000 | 1.0000 | 1.0000 | 0.00 |
| digits_public_feature_head | scored_topk_noisy | k=2,sigma=0.001 | 10 | 5.59e-02 | 0.9686 | 1.0000 | 0.9667 | 0.10 |
| digits_public_feature_head | scored_topk_noisy | k=2,sigma=0.005 | 10 | 5.71e-02 | 0.9686 | 1.0000 | 0.9667 | 0.10 |
| digits_public_feature_head | scored_topk_noisy | k=2,sigma=0.01 | 10 | 5.73e-02 | 0.9681 | 1.0000 | 0.9667 | 0.10 |
| digits_public_feature_head | scored_topk_noisy | k=2,sigma=0.05 | 10 | 6.43e-02 | 0.9644 | 1.0000 | 0.9667 | 0.10 |
| digits_public_feature_head | scored_topk_noisy | k=2,sigma=0.1 | 10 | 7.60e-02 | 0.9553 | 1.0000 | 0.9667 | 0.10 |
| digits_public_feature_head | scored_topk_random_queries | k=2,exact_scores | 10 | 5.58e-02 | 0.9691 | 1.0000 | 0.9667 | 0.10 |
| digits_public_feature_head | scored_topk_rounded | k=2,decimals=0 | 10 | 1.25e-01 | 0.9027 | 1.0000 | 0.9667 | 0.10 |
| digits_public_feature_head | scored_topk_rounded | k=2,decimals=1 | 10 | 6.34e-02 | 0.9628 | 1.0000 | 0.9667 | 0.10 |
| digits_public_feature_head | scored_topk_rounded | k=2,decimals=2 | 10 | 5.63e-02 | 0.9681 | 1.0000 | 0.9667 | 0.10 |
| digits_public_feature_head | scored_topk_rounded | k=2,decimals=3 | 10 | 5.58e-02 | 0.9691 | 1.0000 | 0.9667 | 0.10 |
| digits_public_feature_head | topk_index_only | k=2,no_scores | 10 | n/a | n/a | 1.0000 | 0.0000 | 1.00 |

Interpretation: exact raw or score-bearing APIs expose linear constraints and therefore leak observable adapter parameters. Rounding and noise can preserve labels while increasing parameter error. Scoreless hard-label and top-k-index APIs preserve prediction service but do not support exact linear-system recovery. Endpoint-only PEFT rows report the observable quotient: hidden LoRA effects in the classifier nullspace remain non-identifiable even though endpoint labels are unchanged.
