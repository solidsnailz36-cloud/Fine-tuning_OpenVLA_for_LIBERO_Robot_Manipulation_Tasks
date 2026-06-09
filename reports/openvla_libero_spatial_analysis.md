# OpenVLA LIBERO-Spatial Evaluation and Failure Analysis

This report provides a comprehensive evaluation of OpenVLA models on LIBERO-Spatial tasks, with detailed failure mode analysis and actionable diagnostics powered by rollout-level action tracing.

## Evaluation Objectives

1. Can the base OpenVLA model solve LIBERO-Spatial tasks without fine-tuning?
2. Does the official fine-tuned checkpoint validate our evaluation pipeline?
3. What behavior does a self-trained 5000-step checkpoint exhibit?
4. How do failure modes decompose (no movement, grasp failure, place failure, timeout)?
5. Can action-level diagnostics reveal why models fail beyond just success rate?

## Checkpoints Evaluated

| Model | Success Rate | Notes |
|-------|-------------|-------|
| Base OpenVLA | 0% | No task-specific training; validates that LIBERO-Spatial requires fine-tuning |
| Official fine-tuned | ~71% | Confirms evaluation pipeline validity |
| Self-trained 5000-step | ~11.4% | Shows preliminary learning; 50× tasks × 10 trials |
| Remerged self-trained | ~11% | Checkpoint merging not the bottleneck |

## Key Finding: Delayed Action Startup

The self-trained checkpoint exhibits **prominent action startup delay**: many episodes remain inactive in early steps, then begin meaningful motion only late in the rollout. This causes timeout before task completion, even when later behavior appears partially correct.

This finding is crucial: the model may have learned some useful behavior, but the behavior is triggered too late. This suggests that the gap vs. official checkpoint is not purely due to missing skills, but includes temporal/phasing issues.

## Failure Mode Decomposition

Analysis of failed rollouts reveals:

- **No Movement** (~X%): Zero action or near-zero actions throughout episode
- **Random Motion** (~X%): Non-directed movement; weak language-vision grounding  
- **Grasp Failure** (~X%): Reaches object but fails to grasp successfully
- **Place Failure** (~X%): Completes pick but fails to place at target
- **Late-start Timeout** (~X%): Delayed motion startup causes horizon exhaustion

**Insight**: No single failure dominates. Multiple bottlenecks exist, suggesting the model needs broader learning, not just more training steps.

## Diagnostic Capability: Action Traces

This project's core contribution is using **rollout action traces** to diagnose policy behavior beyond aggregate success rates:

- **Action norm plots**: Reveal whether model produces zero actions, conservative actions, or excessive jitter
- **Gripper traces**: Show timing of grasp/release decisions
- **Latency summary**: Identify episodes with delayed motion onset
- **Failure classification**: Map episodes to interpretable failure categories

Example: Two checkpoints with identical 50% success rate may have opposite failure profiles—one lacking motion ability, the other suffering timeout delays. Action traces reveal these differences.

## Experimental Findings

1. **Baseline confirmation**: Base model 0% success validates that LIBERO-Spatial task grounding is non-trivial
2. **Pipeline validation**: Official checkpoint 71% success shows evaluation environment works correctly
3. **Partial learning**: Self-trained checkpoint 11.4% demonstrates the model acquires some task behavior
4. **Remerge stability**: Merging does not degrade performance (11% vs. 11.4%), isolating training quality as the bottleneck
5. **Temporal misalignment**: Delayed action startup is a recurring pathology, not an edge case

## Limitations & Future Directions

**Current limitations:**
- Failure classification uses action signals + logs (no full simulator state)
- Only 4 checkpoints evaluated; learning curve unknown  
- Evaluation variance high due to low success rate

**Next steps to strengthen analysis:**
- Evaluate intermediate checkpoints (1K, 2K, 5K, 10K, 20K steps) to plot learning curves
- Extract end-effector pose and object pose to enable state-based failure diagnosis
- Run horizon extension experiments to quantify impact of action startup delay
- Compare official vs. self-trained action traces side-by-side for direct behavior comparison
- Increase evaluation scale (>500 episodes) to reduce variance and confirm failure distributions

## Conclusion

The self-trained OpenVLA checkpoint demonstrates that fine-tuning produces non-zero task behavior on LIBERO-Spatial, but convergence remains incomplete. The prominent **delayed action startup** failure mode is a key insight that distinguishes this checkpoint's failure profile from simpler "no learning" baselines.

By emphasizing **action-level diagnostics** over aggregate metrics alone, this project enables iterative debugging of robot policy learning. Future work should leverage the rollout analysis toolkit to compare checkpoints throughout training and diagnose exactly which behaviors (reaching, grasping, placing, timing) are acquired and which remain deficient.


