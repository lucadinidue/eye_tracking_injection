# Extended from https://github.com/huggingface/transformers/blob/main/src/transformers/models/roberta/modeling_roberta.py
"""PyTorch RoBERTa model."""

from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn

from transformers.utils import logging

from transformers import RobertaPreTrainedModel, RobertaModel

from transformers.models.roberta.modeling_roberta import RobertaClassificationHead

logger = logging.get_logger(__name__)


ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "roberta-base",
    "roberta-large",
    "roberta-large-mnli",
    "distilroberta-base",
    "roberta-base-openai-detector",
    "roberta-large-openai-detector",
    # See all RoBERTa models at https://huggingface.co/models?filter=roberta
]


import torch
from torch.nn import MSELoss, L1Loss, CrossEntropyLoss
from dataclasses import dataclass
from typing import Optional, Tuple
from transformers.utils import logging, ModelOutput


def mask_loss(b_output, b_target, target_pad):
    """
    Masks the pad tokens of by setting the corresponding output and target tokens equal.
    """
    active_outputs = b_output.view(-1)
    active_targets = b_target.view(-1)
    active_mask = active_targets == target_pad

    active_outputs = torch.where(active_mask, active_targets, active_outputs)

    return active_outputs, active_targets



@dataclass
class MultiTaskTokenClassifierOutput(ModelOutput):
    """
    Class for outputs of multitask token classification models.

    Args:
        loss (float):
            average MSE loss.
        mse_loss (dict) :
            MSE loss for each task.
        mae_loss (dict) :
            MAE loss for each task.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_labels)`):
            Classification scores (before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the selfattention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    mse_loss: Optional[dict] = None
    mae_loss: Optional[dict] = None
    logits: Tuple[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    labels: Optional[dict] = None


class RobertaForMultiTaskTokenClassification(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)

        # classifiers
        self.tasks = config.tasks
        self.num_tasks = len(self.tasks)
        self.classifiers = nn.ModuleDict({
            task: nn.Linear(config.hidden_size, 1) for
            task in self.tasks
        })

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            token_type_ids: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **labels_list # contiene le label
    ) -> Union[Tuple[torch.Tensor], MultiTaskTokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


        labels = dict()
        for key, value in labels_list.items():
            if key.startswith('label_'):
                labels[key[len('label_'):]] = value
                

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)

        loss = 0
        mse_loss = {}
        mae_loss = {}
        logits = {}

        for task in self.tasks:
            task_logits = self.classifiers[task](sequence_output)
            logits[task] = task_logits

            if labels[task] is not None:
                task_labels = labels[task].to(task_logits.device)
                # TODO: mask out the output associated with not-first-token of a word
                # ERROR: check dimensionalities
                output_, target_ = mask_loss(task_logits, task_labels, -100)

                # MSE Loss
                loss_fct = MSELoss()
                task_mse_loss = loss_fct(output_, target_)
                loss += task_mse_loss

                with torch.no_grad():
                    mse_loss[task] = task_mse_loss
                    # MAE Loss
                    loss_fct = L1Loss()
                    mae_loss[task] = loss_fct(output_, target_)


        loss /= self.num_tasks

        # if not return_dict:                               Se serve bisogna sistemare i logits perchè hanno dimensioni diverse
        #     logits = torch.stack(logits, dim=-1)
        #     output = (logits,) + outputs[2:]
        #     return ((loss,) + output) if loss is not None else output


        return MultiTaskTokenClassifierOutput(
            loss=loss,
            mse_loss=mse_loss,
            mae_loss=mae_loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            labels=labels,
        )
    
class RobertaForMultiTaskTokenClassificationWithWeight(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)

        # classifiers
        self.tasks = config.tasks
        self.num_tasks = len(self.tasks)
        self.classifiers = nn.ModuleDict({
            task: nn.Linear(config.hidden_size, 1) for task in self.tasks
        })

        # Initialize weights and apply final processing
        self.post_init()

        # Trainable weights for task losses
        self.loss_weights = nn.Parameter(torch.ones(self.num_tasks)*(1/self.num_tasks))

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            token_type_ids: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **labels_list  # contains the labels
    ) -> Union[Tuple[torch.Tensor], MultiTaskTokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        labels = dict()
        for key, value in labels_list.items():
            if key.startswith('label_'):
                labels[key[len('label_'):]] = value

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)

        loss = 0
        mse_loss = {}
        mae_loss = {}
        logits = {}

        # Apply softmax to the loss weights
        loss_weights = torch.softmax(self.loss_weights, dim=0)

        for idx, task in enumerate(self.tasks):
            task_logits = self.classifiers[task](sequence_output)
            logits[task] = task_logits

            if labels[task] is not None:
                task_labels = labels[task].to(task_logits.device)
                # TODO: mask out the output associated with not-first-token of a word
                # ERROR: check dimensionalities
                output_, target_ = mask_loss(task_logits, task_labels, -100)

                # MSE Loss
                loss_fct = MSELoss()
                task_mse_loss = loss_fct(output_, target_)
                loss += loss_weights[idx] * task_mse_loss

                with torch.no_grad():
                    mse_loss[task] = task_mse_loss
                    # MAE Loss
                    loss_fct = L1Loss()
                    mae_loss[task] = loss_fct(output_, target_)

        # No need to average the loss since we are weighting it
        # loss /= self.num_tasks

        return MultiTaskTokenClassifierOutput(
            loss=loss,
            mse_loss=mse_loss,
            mae_loss=mae_loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            labels=labels,
        )


class RobertaForInterleavedMultitask(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)

        # token classifiers
        self.token_tasks = config.token_tasks
        self.token_classifiers = nn.ModuleDict({
            task: nn.Linear(config.hidden_size, 1) for task in self.token_tasks
        })

        # sentence classifiers
        self.downstream_task = config.donwstream_task
        self.sentence_classifier = RobertaClassificationHead(config)

        # Initialize weights and apply final processing
        self.post_init()

        # Trainable weights for task losses
        self.num_tasks = len(self.token_tasks) + 1 #last weight is for sentence classification
        self.loss_weights = nn.Parameter(torch.ones(self.num_tasks+1))

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            token_type_ids: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **labels_list  # contains the labels
    ) -> Union[Tuple[torch.Tensor], MultiTaskTokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        labels = dict()
        for key, value in labels_list.items():
            if key.startswith('label_'):
                labels[key[len('label_'):]] = value

        sequence_output = outputs[0]
        loss_weights = torch.softmax(self.loss_weights, dim=0)
        loss = 0
        mse_loss = {}
        mae_loss = {}
        logits = {}

        if list(labels.keys())[0] == self.downstream_task:
            
            logits = self.sentence_classifier(sequence_output)
            labels = labels[self.downstream_task]

            labels = labels.to(logits.device)

            if self.config.downstream_type == "regression":
                loss_fct = MSELoss()
                loss = loss_fct(logits.squeeze(), labels.squeeze())
            elif self.config.downstream_type == "classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            loss =  loss_weights[-1] * loss

        else:
            sequence_output = self.dropout(sequence_output)

            # Apply softmax to the loss weights
            loss_weights = torch.softmax(self.loss_weights, dim=0)

            for idx, task in enumerate(self.token_tasks):
                task_logits = self.token_classifiers[task](sequence_output)
                logits[task] = task_logits
                if labels[task] is not None:
                    task_labels = labels[task].to(task_logits.device)
                    # TODO: mask out the output associated with not-first-token of a word
                    # ERROR: check dimensionalities
                    output_, target_ = mask_loss(task_logits, task_labels, -100)

                    # MSE Loss
                    loss_fct = MSELoss()
                    task_mse_loss = loss_fct(output_, target_)
                    loss += loss_weights[idx] * task_mse_loss

                    with torch.no_grad():
                        mse_loss[task] = task_mse_loss
                        # MAE Loss
                        loss_fct = L1Loss()
                        mae_loss[task] = loss_fct(output_, target_)

        # No need to average the loss since we are weighting it
        # loss /= self.num_tasks

        return MultiTaskTokenClassifierOutput(
            loss=loss,
            mse_loss=mse_loss,
            mae_loss=mae_loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            labels=labels,
        )
