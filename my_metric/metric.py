import json
import re


def do_json_parse(str):
    try:
        return json.loads(regular_json_str(str))
    except Exception:
        # print("json error:", str)
        raise Exception


def do_metric(predict_json_list, label_json_list, half_precision=False, statistic_error_json=False, gc=False):
    tp, fp, fn, error_json, total_json = 0, 0, 0, 0, len(predict_json_list)
    for predict_json, label_json in zip(predict_json_list, label_json_list):
        if not gc:
            try:
                predict_json = do_json_parse(predict_json)
            except Exception:
                error_json = error_json + 1
                if statistic_error_json:
                    continue
                else:
                    raise BaseException
        label_json = do_json_parse(label_json)
        predict_json = regular_json_key(predict_json)
        entity_tp, entity_fp, entity_fn = do_statistic(predict_json["chemical_compounds"], label_json["chemical_compounds"], half_precision)
        property_tp, property_fp, property_fn = do_statistic(predict_json["property_name"], label_json["property_name"], half_precision)
        value_tp, value_fp, value_fn = do_statistic(predict_json["property_value"], label_json["property_value"], half_precision)
        tp = tp + entity_tp + property_tp + value_tp
        fp = fp + entity_fp + property_fp + value_fp
        fn = fn + entity_fn + property_fn + value_fn
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    if statistic_error_json:
        return precision, recall, f1, (error_json / total_json)
    return precision, recall, f1


def regular_json_key(input_json):
    keys = ["chemical_compounds", "property_name", "property_value"]
    for k in keys:
        if k not in input_json:
            input_json[k] = []
    return input_json


def regular_json_str(json_str):
    json_str = re.sub("^\{'", "{\"", json_str)
    json_str = re.sub("': \['", "\": [\"", json_str)
    json_str = re.sub("', '", "\", \"", json_str)
    json_str = re.sub("','", "\",\"", json_str)
    json_str = re.sub("']}", "\"]}", json_str)
    json_str = re.sub("'], '", "\"], \"", json_str)
    json_str = re.sub("': \[]", "\": []", json_str)
    json_str = re.sub("\[], '", "[], \"", json_str)
    # json_str = re.sub("", "", json_str)
    json_str = re.sub("<\|begin_of_text\|>system", "", json_str)
    json_str = re.sub("<\|end_of_text\|>", "", json_str)
    return json_str


def do_statistic(predict_list, label_list, half_precision=False):
    true_positive = 0
    false_positive = 0
    false_negative = 0
    for predict in predict_list:
        if predict in label_list:
            true_positive = true_positive + 1
        else:
            false_positive = false_positive + 1
    for label in label_list:
        if label not in predict_list:
            false_negative = false_negative + 1
    if half_precision:
        for predict in predict_list:
            if predict in label_list:
                continue
            for label in label_list:
                if predict in label:
                    true_positive = true_positive + 0.5
        for label in label_list:
            if label in predict_list:
                continue
            for predict in predict_list:
                if label in predict:
                    false_negative = false_negative - 0.5
    return true_positive, false_positive, false_negative
