sbatch --job-name=boston_housing parallel_benchmark.sh boston_housing
sbatch --job-name=game_of_life parallel_benchmark.sh game_of_life
sbatch --job-name=hpo parallel_benchmark.sh hpo
sbatch --job-name=santafe parallel_benchmark.sh santafe
#sbatch parallel_benchmark_cnn.sh
