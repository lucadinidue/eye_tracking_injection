from transformers import DataCollatorForTokenClassification, DataCollatorWithPadding

class DataCollatorForMultiTaskTokenClassification(DataCollatorForTokenClassification):

    def __call__(self, features):
        return self.torch_call(features)

    def torch_call(self, features):
        import torch

        labels_names = [feat_name for feat_name in features[0].keys() if feat_name.startswith('label_')]
        no_labels_features = [{k: v for k, v in feature.items() if k not in labels_names} for feature in features]

        batch = self.tokenizer.pad(
            no_labels_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        all_labels = dict()
        for label_name in labels_names:
            all_labels[label_name] = [feature[label_name] for feature in features]

        if len(all_labels) == 0:
            return batch

        sequence_length = batch["input_ids"].shape[1]
        padding_side = self.tokenizer.padding_side

        def to_list(tensor_or_iterable):
            if isinstance(tensor_or_iterable, torch.Tensor):
                return tensor_or_iterable.tolist()
            return list(tensor_or_iterable)

        batch_labels_dict = dict()
        if padding_side == "right":
            for label_name in labels_names:
                labels = all_labels[label_name]
                batch_labels_dict[label_name] = [
                    to_list(label) + [self.label_pad_token_id] * (sequence_length - len(label)) for label in labels
                ]
        else:
            for label_name in labels_names:
                labels = all_labels[label_name]
                batch_labels_dict[label_name] = [
                    [self.label_pad_token_id] * (sequence_length - len(label)) + to_list(label) for label in labels
                ]

        # batch['labels'] = dict()
        # for label_name in labels_names:
        #         batch['labels'][label_name[len('label_'):]] = torch.tensor(batch_labels_dict[label_name],
                                                                        #    dtype=torch.float32)

        for label in labels_names:
            batch[label] = torch.tensor(batch_labels_dict[label_name], dtype=torch.float32)

        return batch


class DataCollatorForInterleavedMultiTask(DataCollatorForTokenClassification):
    
    def __init__(self, tokenizer, dst_label, eye_gaze_labels):
        self.eye_gaze_data_collator = DataCollatorForMultiTaskTokenClassification(tokenizer)
        self.dst_data_collator = DataCollatorWithPadding(tokenizer)
        self.dst_label = dst_label
        self.eye_gaze_labels = eye_gaze_labels

    def __call__(self, features):
        return self.torch_call(features)

    def torch_call(self, features):
        if (self.dst_label not in features[0]) or (features[0][self.dst_label] is None): 
            filtered_features = [{k: v for k, v in sentence.items() if k != self.dst_label} for sentence in features]
            batch = self.eye_gaze_data_collator(filtered_features)
        else: 
            filtered_features = [{k: v for k, v in sentence.items() if k not in self.eye_gaze_labels} for sentence in features]
            batch = self.dst_data_collator(filtered_features)
        return batch