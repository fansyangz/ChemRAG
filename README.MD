# ChemRAG

## Requirements

You can install all required Python packages using the provided requirements.txt file using `pip install -r requirements.txt`

## Running the code

The script for fine-tuning by using the following command:

```bash
python tuning/llama3_tune.py \
    --batch_size \
    --local_rank \
    --max_steps \
    --epoch \
    --use_rag 
```
