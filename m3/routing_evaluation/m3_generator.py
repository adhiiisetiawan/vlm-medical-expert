import torch
from huggingface_hub import snapshot_download
from transformers import BitsAndBytesConfig

from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
from llava.constants import IMAGE_TOKEN_INDEX
from llava.utils import disable_torch_init
from llava.conversation import conv_templates

from utils import load_image, get_modality, SwitchLayer

MODEL_CARDS = """Here is a list of available expert models:

<BRATS(args)> 
    Modality: MRI
    Task: segmentation
    Overview: A pre-trained model for volumetric (3D) segmentation of brain tumor subregions from multimodal MRIs based on BraTS 2018 data
    Accuracy: Tumor core (TC): 0.8559 - Whole tumor (WT): 0.9026 - Enhancing tumor (ET): 0.7905 - Average: 0.8518
    Valid args: None

<VISTA3D(args)> 
    Modality: CT
    Task: segmentation
    Overview: domain-specialized interactive foundation model developed for segmenting and annotating human anatomies with precision
    Accuracy: 127 organs: 0.792 Dice on average
    Valid args: 'everything', 'hepatic tumor', 'pancreatic tumor', 'lung tumor', 'bone lesion', 'organs', 'cardiovascular', 'gastrointestinal', 'skeleton', or 'muscles'

<VISTA2D(args)> 
    Modality: cell imaging
    Task: segmentation
    Overview: model for cell segmentation, which was trained on a variety of cell imaging outputs, including brightfield, phase-contrast, fluorescence, confocal, or electron microscopy
    Accuracy: Good accuracy across several cell imaging datasets
    Valid args: None

<CXR(args)> 
    Modality: chest x-ray (CXR)
    Task: classification
    Overview: pre-trained model which are trained on large cohorts of data
    Accuracy: Good accuracy across several diverse chest x-rays datasets
    Valid args: None

Give the model <NAME(args)> when selecting a suitable expert model.
"""

class M3Generator:
    def __init__(self, source="huggingface", model_path="MONAI/Llama3-VILA-M3-8B", conv_mode="llama_3"):
        """Initialize the M3 generator"""
        global SYS_PROMPT
        self.source = source

        # conversation template 
        self.conv = conv_templates[conv_mode].copy()
        self.user_role = self.conv.roles[0]
        self.assistant_role = self.conv.roles[1]

        # tensor cache 
        self.image_tensor_cache = {}


        if source == "local" or source == "huggingface":
            # TODO: allow setting the device
            disable_torch_init()
            self.conv_mode = conv_mode
            if source == "huggingface":
                model_path = snapshot_download(model_path)
            model_name = get_model_name_from_path(model_path)

            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

            self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
                model_path,
                model_name,
                device_map="auto",
                quantization_config=quant_config,
                max_memory={
                    0: "40GiB"
                }
            )

            self.model.config.use_cache = False
            print("Device:", next(self.model.parameters()).device)
            print("allocated:", torch.cuda.memory_allocated() / 1024**3, "GB")
            print("reserved:", torch.cuda.memory_reserved() / 1024**3, "GB")

            SYS_PROMPT = conv_templates[self.conv_mode].system  # only set once
        else:
            raise NotImplementedError(f"Source {source} is not supported.")
        
    # Switch Layer
    def insert_switch_layer(self, 
                            text_expert_count:int=4, text_top_k:int=2, 
                            text_expert_period:int=5, text_expert_offset:int=3,
                            vision_expert_count:int=4, vision_top_k:int=2, 
                            vision_expert_period:int=5, vision_expert_offset:int=3
                            ):
        """
        Inserts switch layer into the model.

        Parameters
        ------------
        text_expert_count (int) : Amount of available experts on each switch layer for text model
        text_top_k (int) : Total amount of expert used for a single token for text model
        text_expert_period (int) : Period of the switch layer occurence for text model
        text_expert_offset (int) : Offset of the switch layer occurence for text model

        vision_expert_count (int) : Amount of available experts on each switch layer for vision model
        vision_top_k (int) : Total amount of expert used for a single token for vision model
        vision_expert_period (int) : Period of the switch layer occurence for vision model
        vision_expert_offset (int) : Offset of the switch layer occurence for vision model

        """
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        self.switch_params = {
            "text_expert_count": text_expert_count,
            "text_top_k": text_top_k,
            "text_expert_period": text_expert_period,
            "text_expert_offset": text_expert_offset,
            "vision_expert_count": vision_expert_count,
            "vision_top_k": vision_top_k,
            "vision_expert_period": vision_expert_period,
            "vision_expert_offset": vision_expert_offset,
        }

        for blocks in self.model.llm.model.layers[text_expert_offset::text_expert_period]:
            switch = SwitchLayer(2560, blocks.mlp, text_expert_count, text_top_k).to(DEVICE)
            blocks.mlp = switch

        for blocks in self.model.vision_tower.vision_tower.vision_model.encoder.layers[vision_expert_offset::vision_expert_period]:
            switch = SwitchLayer(1152, blocks.mlp, vision_expert_count, vision_top_k).to(DEVICE)
            blocks.mlp = switch

        print("Switch layer successfully implemented")
        print("Device:", next(self.model.parameters()).device)
        print("allocated:", torch.cuda.memory_allocated() / 1024**3, "GB")
        print("reserved:", torch.cuda.memory_reserved() / 1024**3, "GB")
    
    def get_expert_trigger_count(self):
        text = []
        vision = []
        for blocks in self.model.llm.model.layers[self.switch_params["text_expert_offset"]::self.switch_params["text_expert_period"]]:
            text.append(blocks.mlp.get_expert_count())

        for blocks in self.model.vision_tower.vision_tower.vision_model.encoder.layers[self.switch_params["vision_expert_offset"]::self.switch_params["vision_expert_period"]]:
            vision.append(blocks.mlp.get_expert_count())

        return {"text": text, "vision":vision}

        
        
    # IMAGE HANDLING
    def _get_image_tensor(self, image_path):
        """Load + preprocess image ONCE."""

        if image_path in self.image_tensor_cache:
            return self.image_tensor_cache[image_path]

        img = load_image(image_path)

        tensor = process_images(
            [img],
            self.image_processor,
            self.model.config,
        ).to(self.model.device, dtype=torch.float16)

        self.image_tensor_cache[image_path] = tensor
        return tensor

    # PROMPT BUILD
    def _build_prompt(self, user_text):
        """Minimal llama conversation prompt."""

        conv = self.conv.copy()
        conv.append_message(self.user_role, user_text)
        conv.append_message(self.assistant_role, "")
        return conv.get_prompt(), conv.sep

    # GENERATION
    @torch.inference_mode()
    def generate(
        self,
        prompt,
        image_paths=None,
        max_new_tokens=512,
    ):
        """
        Generate response.

        Parameters
        ------------
        prompt : str
        image_paths : list[str] | None
        """
        modality = get_modality(image_paths, text=prompt)
        if isinstance(image_paths, list):
            # multi-modal images
            prompt = (
                prompt.replace("<image>", "") if "<image>" in prompt else prompt
            )  # remove the image token if it's in the prompt
            special_token = "T1(contrast enhanced): <image>, T1: <image>, T2: <image>, FLAIR: <image>. "
            mod_msg = f"These are different {modality} modalities."
            _prompt = f"""\
Reply to the prompt below by answering or calling an expert model. Only call an expert model if absolutely necessary.

Prompt: {prompt}
{special_token + mod_msg}

{MODEL_CARDS}
"""

        prompt_text, stop_str = self._build_prompt(_prompt)

        # tokenize
        input_ids = tokenizer_image_token(
            prompt_text,
            self.tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors="pt",
        ).unsqueeze(0).to(self.model.device)

        # images
        images_input = None
        if image_paths:
            tensors = [self._get_image_tensor(p) for p in image_paths]
            images_input = tensors

        # generate
        output_ids = self.model.generate(
            input_ids,
            images=images_input,
            do_sample=False,
            num_beams=1,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        text = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        if text.endswith(stop_str):
            text = text[:-len(stop_str)].strip()

        return text, _prompt

    # UTILITY

    def clear_image_cache(self):
        self.image_tensor_cache.clear()

    def cache_size(self):
        return len(self.image_tensor_cache)
