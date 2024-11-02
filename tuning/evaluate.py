from transformers.trainer_callback import DefaultFlowCallback, TrainerState, TrainerControl
from transformers import TrainingArguments
import os, sys
def check_env():
    current_env = "/ai/DL/zz"
    r = os.system(f"cd {current_env}")
    current_env = current_env if r == 0 else "/data/zhouyangfan/second"
    sys.path.append(current_env)
    return current_env
current_env = check_env()
from tuning.llama3_evaluation import do_evaluation
from tuning.metric import metric_by_outputs, metric_by_result


class EvaluateCallable(DefaultFlowCallback):
    def __init__(self, trainer, validate_iter=0, test_tokenizer=None, use_rag=False):
        super().__init__()
        self.trainer = trainer
        self.validate_iter = validate_iter
        self.test_tokenizer = test_tokenizer
        self.use_rag=use_rag

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if self.validate_iter != 0 and state.global_step != 1 and (state.global_step % self.validate_iter == 0):
            infer_result_list = do_evaluation(self.trainer.model, self.test_tokenizer, use_rag=self.use_rag)
            precision, recall, f1, error_json_ratio = metric_by_result(infer_result_list, gc=True)
            print(f"-------------step: {str(state.global_step)}-------------")
            print(f"-------------precision: {str(precision)}, recall: {str(recall)}, f1: {str(f1)}, "
                  f"error_json_ratio: {str(error_json_ratio)}-------------")