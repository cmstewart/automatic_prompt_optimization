# Automatic Prompt Optimization with RAG                                                                                                                                                    

This repository compares three evaluator strategies for ProTeGi-style automatic prompt optimization in retrieval-augmented financial question answering:                                    - **UCB** (Upper Confidence Bound) - multi-armed bandit selection
- **PPO** (Proximal Policy Optimization) - reinforcement learning                                                                                                                           
- **DPO** (Direct Preference Optimization) - preference learning
                                                                                                                                                                                            The system iteratively generates candidate prompts, scores them against RAG-based QA samples using BEM evaluation, and selects the best candidates across multiple optimization rounds. Experiments are run across three financial QA datasets (FinanceBench, FinQA, FinDoc-RAG) at three compute budget levels.
                      
## Key Results                                                                                                                                                                                                
Through one round of experimentation, UCB achieves the highest final accuracy at large evaluation budgets, while DPO is competitive at moderate budgets. PPO requires larger budgets to perform reliably. Budget level is a stronger predictor of optimization quality than evaluator choice. See `results/analysis.ipynb` for the full analysis. T

## Repository Structure                                                                                                                                                                     - `main.py` - Optimization loop with per-round checkpointing                                                                                                                                - `run_experiments.py` - Batch runner supporting multiple datasets, evaluators, budgets, and random seeds
- `prepare_data.py`, `vectorize.py` - Dataset preparation and Chroma vectorstore construction
- `predictors.py`, `optimizers.py`, `evaluators.py`, `scorers.py` - Core pipeline components                                                                                                - `results/` - Experiment outputs and analysis notebook
                      
## Running Experiments                                    
Experiments are designed to run on GCP VMs. See `setup_vm.sh` for environment setup. The runner supports flexible experiment selection:

`python run_experiments.py --dataset finqa --evaluators ucb ppo dpo --budgets 1 2 3 --seeds 1 2 3`
Checkpointing, rate limit handling, and automatic skip of completed runs are built in.                                                                                                                        
        
