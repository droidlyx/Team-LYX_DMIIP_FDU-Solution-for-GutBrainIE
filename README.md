# Team-LYX_DMIIP_FDU-Solution-for-GutBrainIE

## Steps to reproduce

First, download three models BiomedBERT-base-uncased-abstract-fulltext, BioLinkBERT-base, xlm-roberta-large-english-clinical from Huggingface and put them in base_models folder.

Then, put GutBrainIE train data into data/train folder

Finally, in the project base folder, run the following commands (I used RTX4090 GPU with 24GB GPU memory):

python data2pub.py

python train_NER.py --base_model_path './base_models/BiomedBERT-base-uncased-abstract-fulltext'

python train_NER.py --base_model_path './base_models/BioLinkBERT-base'

python train_NER.py --base_model_path './base_models/xlm-roberta-large-english-clinical'

python run_NER.py --test_path './data/test/' --merge_all_model_outputs True --model_names BioLinkBERT_base BiomedBERT_base_uncased_abstract_fulltext xlm_roberta_large_english_clinical

python train_classifier.py --base_model_path "./base_models/BioLinkBERT-base"

python get_relations.py --re_model_path './finetuned_models/finetuned_classifiers/BioLinkBERT-base/' --input_path './results/xxx_merged/' (replace xxx with the folder name in results)

python pub2jsonl.py --input_path './results/xxx_merged/predictions/'
