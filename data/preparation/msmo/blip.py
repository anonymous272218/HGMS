from transformers import BlipProcessor, BlipForImageTextRetrieval, BlipConfig, BlipVisionModel, BlipTextModel, \
    BlipPreTrainedModel
from torch import nn
from torch.nn.functional import normalize
import torch
from transformers.utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from typing import Any, Optional, Tuple, Union

from dataclasses import dataclass


@dataclass
class BlipOutput(ModelOutput):
    itm_score: Optional[torch.Tensor] = None
    text_feature: Optional[torch.Tensor] = None
    vision_feature: Optional[torch.Tensor] = None


class Blip(BlipForImageTextRetrieval):
    config_class = BlipConfig

    def __init__(self, config: BlipConfig):
        super().__init__(config)

    def forward(
            self,
            input_ids: torch.LongTensor,
            pixel_values: torch.FloatTensor,
            use_itm_head: Optional[bool] = True,
            attention_mask: Optional[torch.LongTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BlipOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        image_embeds = vision_outputs[0]
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long)

        itm_output = []

        for i in range(len(image_embeds)):
            question_embeds = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                encoder_hidden_states=image_embeds[i:i + 1],
                encoder_attention_mask=image_atts[i:i + 1],
                return_dict=return_dict,
            )
            question_embeds = question_embeds[0] if not return_dict else question_embeds.last_hidden_state
            itm_output.append(self.itm_head(question_embeds[:, 0, :]))
        itm_output = torch.stack(itm_output)

        vision_feature = self.vision_proj(image_embeds[:, 0, :])

        return BlipOutput(
            itm_score=itm_output,
            vision_feature=vision_feature,
        )


import torch
from torch.utils.data import Dataset
from PIL import Image


class BlipDataset(Dataset):
    def __init__(self, data_list, hash2img, blip_model_path):
        self.data = data_list
        self.hash2img = hash2img
        self.processor = BlipProcessor.from_pretrained(blip_model_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        article = self.data[idx]
        hash = article['hash']
        texts = article['text']
        try:
            images = [Image.open(file) for file in self.hash2img[hash]]
            inputs = self.processor(text=texts[:50], images=images[:16], return_tensors="pt", padding=True,
                                    truncation=True)
        except Exception as e:
            print('error: ' + hash)
            raise e
        return hash, inputs

class BlipSumDataset(Dataset):
    def __init__(self, data_list, hash2img, blip_model_path):
        self.data = data_list
        self.hash2img = hash2img
        self.processor = BlipProcessor.from_pretrained(blip_model_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        article = self.data[idx]
        hash = article['hash']
        summary = ".\n".join(article['summary'])
        try:
            images = [Image.open(file) for file in self.hash2img[hash]]
            inputs = self.processor(text=summary, images=images[:16], return_tensors="pt", padding=True,
                                    truncation=True)
        except Exception as e:
            print('error: ' + hash)
            raise e
        return hash, inputs

def collate_fn(batch):
    hashes = [e[0] for e in batch]
    inputs = [e[1] for e in batch]
    return hashes, inputs
