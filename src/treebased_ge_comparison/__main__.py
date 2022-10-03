import sys
import logging 

sys.path.append('GeneticEngine/')


from src.treebased_ge_comparison.gengine_evaluation import evaluate_geneticengine

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    
    # --mode=generations or --mode=timer
    if len(sys.argv) < 2 or '--mode=' not in sys.argv[1]:
        raise Exception('The --mode=generations or --mode=timer should be included after ./run_ponyge_comparison.sh')
    mode = sys.argv[1].split('=')[1]
    
        
    examples = list()
    folder_addition = ''
    if len(sys.argv) > 2:
        if '--folder_addition=' in sys.argv[2]:
            folder_addition = sys.argv[2].split('=')[1]
            if len(sys.argv) > 2:
                examples = sys.argv[3:]
        else:
            examples = sys.argv[2:]
            
    #evaluate_ponyge(examples)
    evaluate_geneticengine(examples, mode, folder_addition)