# Automatic Prompt Optimization with RAG                                                                                                                                                    

ProTeGi-style automatic prompt optimization for retrieval-augmented financial question answering. The system generates candidate prompts, evaluates them against document retrieval + QA samples using BEM scoring, and iteratively improves them over multiple rounds.                                                                                                              


## Datasets                                                                                                                                                                                            
- **FinanceBench** - Financial document QA with PDF-based retrieval                                                                                                                                           
- **FinQA** - Financial QA with structured context (pre_text, table, post_text)                                                                                                                               
- **FinDoc-RAG** - Financial document QA with markdown reference files (600 QA pairs from 46 banking documents)


## Evaluator Strategies                                   

Three evaluator strategies are compared across different compute budgets:                                                                                                                      
- **UCB** (Upper Confidence Bound) - Multi-armed bandit approach to prompt selection                                                                                                        - **PPO** (Proximal Policy Optimization) - Reinforcement learning-based evaluator
- **DPO** (Direct Preference Optimization) - Preference learning-based evaluator
                                          

## Project Structure

main.py              # Optimization loop with per-round checkpointing
prepare_data.py      # Dataset preparation (parquet generation)                                                                                                                                               
vectorize.py         # Chroma vectorstore builder and retriever factory
generate.py          # Answer generation using optimized prompts
evaluate.py          # Answer grading (GPT or BEM judge)
predictors.py        # QA inference with retrieval                                                                                                                                                            
optimizers.py        # ProTeGi-style prompt expansion and gradient generation
evaluators.py        # UCB, PPO, DPO evaluator implementations                                                                                                                                                
scorers.py           # BEM scoring                        
tasks.py             # Dataset loading and evaluation harness                                                                                                                                                 
utils.py             # OpenAI SDK wrapper
paths.py             # Central path resolver (APO_ROOT env var)                                                                                                                                               
run_experiments.py   # Batch experiment runner with completion tracking
setup_vm.sh          # GCP VM setup script
results/             # Experiment output files and analysis notebook                                                                                                                        
