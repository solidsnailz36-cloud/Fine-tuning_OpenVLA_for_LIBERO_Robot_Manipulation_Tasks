# OpenVLA LIBERO-Spatial Evaluation and Failure Analysis

## 1. Goal

This report summarizes the evaluation and failure analysis of OpenVLA models on the LIBERO-Spatial benchmark.

The goal of this experiment is to understand:

1. Whether the original base OpenVLA model can directly solve LIBERO-Spatial tasks.
2. Whether the official fine-tuned checkpoint can reproduce reasonable benchmark performance.
3. Whether our self-trained 5000-step checkpoint has acquired useful manipulation behavior.
4. Whether checkpoint remerge affects evaluation performance.
5. What types of failures dominate the self-trained model.
6. Whether rollout-level action traces can reveal diagnostic patterns such as delayed motion, grasp failure, place failure, or timeout.

The overall purpose is not only to obtain a success rate, but also to diagnose why the self-trained model underperforms the official fine-tuned checkpoint.

---

## 2. Setup

### 2.1 Environment

The evaluation was conducted in a containerized environment with the OpenVLA project and LIBERO benchmark installed.

Conda environment:
/root/gpufree-data/conda_envs/openvla

Project directory:
/root/gpufree-data/project

Main evaluation code path:
openvla/experiments/robot/libero/run_libero_eval.py

Analysis scripts directory:
scripts/

The final analysis outputs are saved under:
results/

A recommended per-run output structure is:
results/
  run_YYYY_MM_DD-HH_MM_SS/
    results_summary.csv
    latency_summary.csv
    failure_cases.csv
    action_norm_plots/
    gripper_plots/
    
Final report path:
reports/openvla_libero_spatial_analysis.md


## 3. Checkpoints
The following model checkpoints were compared.

### 3.1 Base OpenVLA Model
The base OpenVLA model was evaluated directly on LIBERO-Spatial without task-specific fine-tuning.

Purpose:

Test whether the general OpenVLA model can solve LIBERO-Spatial out of the box.
Establish a lower-bound baseline.

Result:

Success rate: 0%

Interpretation:

The base model failed completely on LIBERO-Spatial, suggesting that this benchmark requires domain-specific fine-tuning. General visual-language-action pretraining alone is insufficient for this task suite.

### 3.2 Official Fine-tuned Checkpoint
The official LIBERO-Spatial fine-tuned checkpoint was evaluated to validate the evaluation pipeline.

Result:

Success rate: approximately 71%

Interpretation:

This result shows that the evaluation pipeline is functioning correctly. Since the official checkpoint achieves a reasonable success rate, the low performance of the base model and self-trained checkpoint is unlikely to be caused by a broken evaluator or environment setup.

### 3.3 Self-trained 5000-step Checkpoint

The self-trained checkpoint was evaluated after 5000 training steps.

Example checkpoint path:
/root/gpufree-data/project/datadisk/checkpoints/openvla-7b+libero_spatial_no_noops+b16+lr-0.0005+lora-r16+dropout-0.0--image_aug

Evaluation scale:

Task suite: LIBERO-Spatial

Number of episodes: 500

Horizon setting: default / 1x horizon

Result:
Success rate: approximately 11.4%

Interpretation:
The self-trained 5000-step checkpoint shows non-zero task performance. This means the model has started to learn some meaningful manipulation behavior. However, its performance remains far below the official fine-tuned checkpoint, indicating that the model is still under-trained or affected by training, data, or optimization limitations.

### 3.4 Remerged Self-trained Checkpoint
A remerged version of the self-trained checkpoint was also evaluated.

Result:

Success rate: approximately 11%

Interpretation:
The remerged checkpoint performs similarly to the original self-trained checkpoint. This suggests that checkpoint merging is not the main reason for the performance gap. The main issue is more likely related to training quality, amount of training, policy behavior, action timing, or task grounding.

## 4. Evaluation Protocol
The evaluation was performed on the LIBERO-Spatial benchmark using the OpenVLA LIBERO evaluation script.

A representative evaluation command is:


HF_ENDPOINT=https://hf-mirror.com python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint /root/gpufree-data/project/datadisk/checkpoints/openvla-7b+libero_spatial_no_noops+b16+lr-0.0005+lora-r16+dropout-0.0--image_aug \
  --task_suite_name libero_spatial \
  --center_crop True \
  --num_trials_per_task 50

  
The evaluation logs are stored under:
openvla/experiments/logs/

Example log directory:
openvla/experiments/logs/EVAL-libero_spatial-openvla-2026_06_04-09_55_28--self_ft_5000_horizon_1x

The evaluation output contains per-task and per-episode rollout information. These logs are then processed by custom analysis scripts to summarize success rates, action traces, gripper behavior, and failure types.

## 5. Main Results
The main results are summarized below.

Model / Checkpoint	LIBERO-Spatial Success Rate	Notes
Base OpenVLA	0%	No task-specific fine-tuning; unable to solve LIBERO-Spatial
Official fine-tuned checkpoint	~71%	Confirms that the evaluation pipeline is valid
Self-trained 5000-step checkpoint	~11.4%	Learns preliminary behavior but remains far below official checkpoint
Remerged self-trained checkpoint	~11%	Similar to pre-remerge result; merge is not the main issue

Key observation:
The self-trained checkpoint is clearly better than the base model, but still far from the official fine-tuned checkpoint. This indicates partial learning rather than complete failure.

## 6. Rollout Analysis Scripts
To better understand model behavior beyond aggregate success rate, we implemented a lightweight rollout analysis toolkit under:

scripts/

The analysis scripts are:

scripts/summarize_results.py
scripts/analyze_rollout_actions.py
scripts/classify_failures.py
scripts/plot_action_traces.py
scripts/rollout_analysis_utils.py

### 6.1 summarize_results.py
Purpose:
Parse evaluation logs.
Aggregate task-level and overall success statistics.
Export summary results to CSV.

Example usage:

python scripts/summarize_results.py \
  --log_root openvla/experiments/logs/EVAL-libero_spatial-openvla-2026_06_04-09_55_28--self_ft_5000_horizon_1x \
  --output results/results_summary.csv
  
Main output:
results/results_summary.csv

This file records the main success-rate statistics used in the report.

### 6.2 analyze_rollout_actions.py
Purpose:
Analyze rollout action sequences.
Compute action magnitude statistics.
Identify episodes with abnormal action behavior.
Help diagnose whether the model is moving immediately, moving late, or barely moving.

Example usage:

python scripts/analyze_rollout_actions.py \
  --log_root openvla/experiments/logs/EVAL-libero_spatial-openvla-2026_06_04-09_55_28--self_ft_5000_horizon_1x \
  --output results/latency_summary.csv
  
Main output:
results/latency_summary.csv

This file is used to inspect delayed motion and action-start behavior.

### 6.3 classify_failures.py
Purpose:
Classify failed episodes into coarse failure modes.
Provide a more interpretable view of why rollouts fail.
The current version uses available rollout-level information and action statistics to classify failures into categories such as:

no_movement
random_motion
grasp_failure
place_failure
late_start_timeout
unknown_failure

Example usage:

python scripts/classify_failures.py \
  --log_root openvla/experiments/logs/EVAL-libero_spatial-openvla-2026_06_04-09_55_28--self_ft_5000_horizon_1x \
  --output results/failure_cases.csv
  
Main output:
results/failure_cases.csv

This file provides per-episode failure labels and is used for the failure mode analysis section.

### 6.4 plot_action_traces.py
Purpose:
plot_action_traces.py visualizes step-level action traces generated during policy evaluation.
It is used to diagnose rollout behavior beyond the final success rate, with a focus on action magnitude, gripper prediction, and delayed movement.

The script parses the evaluation trace directory:
EVAL-xxx/
  traces/
    task_00/
      episode_000/
        trace.jsonl
        meta.json
and generates per-episode plots for action norm and gripper command.

Example usage:

python scripts/plot_action_traces.py \
  --log_root openvla/experiments/logs/EVAL-libero_spatial-openvla-2026_06_04-09_55_28--self_ft_5000_horizon_1x \
  --output_dir results
  
Main outputs:
results/combined_action_gripper_plots/

### 6.5 Recommended One-command Analysis Workflow
To avoid overwriting previous results, each analysis run should be saved into a new timestamped directory:

RUN_DIR=results/run_$(date +"%Y_%m_%d-%H_%M_%S")
mkdir -p "$RUN_DIR"

python scripts/summarize_results.py \
  --log_root openvla/experiments/logs/EVAL-libero_spatial-openvla-2026_06_04-09_55_28--self_ft_5000_horizon_1x \
  --output "$RUN_DIR/results_summary.csv"

python scripts/analyze_rollout_actions.py \
  --log_root openvla/experiments/logs/EVAL-libero_spatial-openvla-2026_06_04-09_55_28--self_ft_5000_horizon_1x \
  --output "$RUN_DIR/latency_summary.csv"

python scripts/classify_failures.py \
  --log_root openvla/experiments/logs/EVAL-libero_spatial-openvla-2026_06_04-09_55_28--self_ft_5000_horizon_1x \
  --output "$RUN_DIR/failure_cases.csv"

python scripts/plot_action_traces.py \
  --log_root openvla/experiments/logs/EVAL-libero_spatial-openvla-2026_06_04-09_55_28--self_ft_5000_horizon_1x \
  --output_dir "$RUN_DIR"
This produces a clean output folder for each analysis run.

## 7. Failure Mode Analysis
The self-trained 5000-step checkpoint shows several dominant failure patterns.

### 7.1 No Movement
Some episodes show little or no effective movement. In these cases, the policy may produce near-zero actions or fail to initiate meaningful robot motion.

This failure mode suggests that the model has not fully learned when and how to begin manipulation in some task states.

Possible causes:

Insufficient training steps.
Weak action supervision early in the trajectory.
Distribution mismatch between training and evaluation states.
Overly conservative action predictions.

### 7.2 Random Motion
In some failed episodes, the robot moves but the motion does not appear directed toward the target object.

This suggests weak visual grounding or task grounding. The policy may produce non-zero actions, but those actions are not aligned with the object specified by the language instruction.

Possible causes:

Insufficient grounding between language and spatial object location.
Limited training data coverage.
Incomplete convergence.
Action prediction noise.

### 7.3 Grasp Failure
Some episodes show behavior where the robot appears to approach the object but fails to grasp it successfully.

This indicates that the model may have partially learned reaching behavior but not precise grasp execution.

Possible causes:

Insufficient fine-grained manipulation learning.
Gripper timing error.
End-effector pose error near the object.
Short or unstable approach phase.
Inadequate training duration.

### 7.4 Place Failure
In some rollouts, the robot appears to interact with or move the object but fails to place it at the target location.

This indicates that the model may partially complete the pick phase but fail during the place phase.

Possible causes:

Incomplete long-horizon behavior learning.
Weak spatial relation understanding.
Accumulated error after grasping.
Poor target placement alignment.
Timeout before successful placement.

### 7.5 Late-start Timeout
A prominent behavior observed in the self-trained checkpoint is action startup delay.

In some episodes, the robot remains inactive or moves weakly during the early part of the rollout, then starts taking more meaningful actions only in the later part of the episode. Because LIBERO evaluation has a finite horizon, this delayed behavior can cause the model to run out of time before completing the task.

This failure mode is especially important because it suggests that the model may have learned some useful behavior, but the behavior is not triggered early enough.

Possible causes:

The model requires more context steps before producing confident actions.
The model learned delayed trajectories from the training distribution.
The action distribution is biased toward low-magnitude actions at the beginning.
The evaluation horizon is too short for the learned behavior pattern.
The self-trained checkpoint has not converged sufficiently.

## 8. Horizon Extension Diagnostic
To test whether delayed behavior is a major bottleneck, a horizon extension diagnostic was performed or considered.

The idea is simple:

If the model often starts moving late, increasing the evaluation horizon may improve success rate.
If increasing the horizon does not improve success rate, then failures are more likely caused by incorrect manipulation behavior rather than insufficient time.
The self-trained checkpoint was evaluated under the standard horizon setting, where it achieved about 11.4% success.

A horizon extension test can be used to answer:

Does the model eventually perform the correct behavior if given more time?
Are failures caused by timeout or by incorrect action generation?
Does late-start behavior explain a significant portion of failures?
Current diagnostic observation:

The self-trained model shows clear action startup delay in some episodes. Some rollouts only begin approaching the object in the later stage, leading to timeout before task completion.

Interpretation:

Horizon extension may improve some episodes, but it is unlikely to fully close the gap to the official fine-tuned checkpoint. The model likely needs better training, not only longer evaluation time.

## 9. Key Findings
The main findings are:

The base OpenVLA model achieves 0% success on LIBERO-Spatial.

This shows that LIBERO-Spatial requires task-specific fine-tuning. General OpenVLA pretraining alone is not sufficient.

The official fine-tuned checkpoint reaches approximately 71% success.

This validates the evaluation pipeline. The environment, task suite, and evaluator are able to produce reasonable results when a strong checkpoint is used.

The self-trained 5000-step checkpoint reaches approximately 11.4% success over 500 episodes.

This indicates that the model has acquired preliminary task behavior, but it is still far below the official fine-tuned checkpoint.

The remerged checkpoint reaches approximately 11% success, close to the pre-remerge result.

Therefore, checkpoint merging is not the main bottleneck.

The self-trained model shows obvious delayed action startup.

Some episodes only begin meaningful motion late in the rollout, causing timeout even when the later behavior appears partially correct.

Failure modes are mainly concentrated in no movement, grasp failure, place failure, and late-start timeout.

These failures suggest that the model has not yet learned stable full-horizon manipulation behavior.

Rollout action analysis is useful for diagnosing policy behavior.

Aggregate success rate alone is insufficient. Action norm plots, gripper traces, latency summaries, and failure classification provide important evidence about how and why the model fails.

## 10. Limitations
This analysis has several limitations.

### 10.1 Coarse Failure Classification
The current failure classification is based mainly on available rollout logs and action-level signals. It does not yet fully use simulator state such as:

End-effector pose
Object pose
Distance between end-effector and target object
Distance between object and goal
Whether the object was lifted
Whether the object was placed near the correct target
Therefore, the failure labels are useful but still approximate.

### 10.2 Limited Training Checkpoints
The current analysis focuses mainly on the base model, official checkpoint, self-trained 5000-step checkpoint, and remerged checkpoint.

More intermediate checkpoints would be useful to understand the learning curve, for example:

1000 steps
2000 steps
5000 steps
10000 steps
20000 steps
This would help determine whether the self-trained model is steadily improving or has plateaued.

### 10.3 Limited Observation-level Diagnostics
The current rollout analysis primarily uses action traces and success/failure metadata. It does not yet perform full simulator-state-based analysis.

Adding observation-level information could improve failure diagnosis, especially for distinguishing:

Random motion vs. directed approach
Grasp failure vs. failure to reach object
Place failure vs. object never grasped
Late-start success potential vs. irrecoverable failure
However, this would require additional engineering to extract and standardize object pose and end-effector pose across LIBERO tasks.

### 10.4 Evaluation Variance
The self-trained model has relatively low success rate, so measured performance may vary across random seeds, initial states, and task subsets.

A larger number of evaluation episodes helps reduce variance, but repeated runs would provide more reliable estimates.

## 11. Next Steps
Recommended next steps are:

### 11.1 Continue Training
The self-trained 5000-step checkpoint shows preliminary behavior but is far below the official checkpoint. The most direct next step is to continue training for more steps.

Recommended checkpoints to evaluate:

10000 steps
20000 steps
50000 steps

The goal is to determine whether success rate continues improving with more training.

### 11.2 Track Learning Curve
Evaluate multiple checkpoints using the same protocol and plot success rate versus training steps.

Example table:

Checkpoint	Success Rate
1000 steps	TBD
5000 steps	~11.4%
10000 steps	TBD
20000 steps	TBD
Official checkpoint	~71%

This would make it easier to determine whether the self-training setup is healthy.

### 11.3 Improve Failure Diagnosis If Needed
If more precise failure analysis becomes necessary, future logs can include:

End-effector pose
Target object pose
Goal object pose
Distance from end-effector to target object
Distance from target object to goal
Object height
Gripper state
These signals would allow more accurate classification of:

random_motion
grasp_failure
place_failure
late_start_timeout

For the current stage, the lightweight action-based analysis is sufficient and avoids unnecessary engineering complexity.

### 11.4 Investigate Delayed Action Startup
Because delayed motion is a clear issue, future work should examine:

Whether early actions are too small.
Whether gripper actions are delayed.
Whether the model waits too long before approaching the object.
Whether horizon extension improves success rate.
Whether training data contains many slow-start trajectories.
Potential fixes include:

More training.
Adjusting action normalization.
Checking no-op filtering.
Inspecting the first 20 to 50 actions of successful and failed rollouts.
Comparing self-trained action traces with official checkpoint action traces.

### 11.5 Compare Against Official Checkpoint Rollouts
A useful diagnostic is to run the same action analysis scripts on the official fine-tuned checkpoint logs.

This would allow direct comparison between:

Official checkpoint action norm traces
Self-trained checkpoint action norm traces
Official gripper timing
Self-trained gripper timing
Success and failure trajectory patterns
This comparison may reveal whether the self-trained model is too slow, too conservative, too noisy, or misaligned in gripper control.

## 12. Conclusion
The evaluation shows a clear performance hierarchy:

Base OpenVLA: 0%
Self-trained 5000-step checkpoint: ~11.4%
Remerged self-trained checkpoint: ~11%
Official fine-tuned checkpoint: ~71%
The base model fails completely, confirming that LIBERO-Spatial requires domain-specific fine-tuning. The official fine-tuned checkpoint achieves strong performance, validating the evaluation setup.

The self-trained 5000-step checkpoint has learned some useful behavior, as shown by its non-zero success rate and partial manipulation attempts. However, its performance remains much lower than the official checkpoint. Remerging does not significantly change the result, suggesting that the main bottleneck is not checkpoint merging.

Failure analysis indicates that the self-trained model suffers from no movement, grasp failure, place failure, and late-start timeout. In particular, delayed action startup appears to be an important diagnostic signal. The rollout analysis scripts provide a useful lightweight toolkit for understanding these behaviors beyond aggregate success rate.

Overall, the self-trained checkpoint is promising but under-trained. The most important next step is to continue training, evaluate more checkpoints, compare action traces against the official checkpoint, and only add more detailed observation-level diagnostics if finer-grained failure attribution becomes necessary.
