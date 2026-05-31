def load_model(model_name: str, model_path: str, load_4bit: bool = False):
    """
    Load vision-language models and associated components based on model name.
    
    Args:
        model_name (str): Name of the model ('llava2', 'mplug_owl2', etc.)
        model_path (str): Path or hub name of the pretrained model.
        load_4bit (bool): Whether to load the model in 4-bit quantization.

    Returns:
        A dictionary containing loaded components: model, tokenizer, processor/image_processor
    """
    model_name = model_name.lower()

    if model_name == "llava-1.5-7b-hf":
        from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
        import torch

        if load_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16
            )
            model = LlavaForConditionalGeneration.from_pretrained(
                model_path,
                attn_implementation="eager",
                quantization_config=quantization_config,
                device_map="auto"
            )
        else:
            model = LlavaForConditionalGeneration.from_pretrained(
                model_path,
                attn_implementation="eager"
            ).cuda()
        processor = AutoProcessor.from_pretrained(model_path)
        tokenizer = processor.tokenizer

        return model, tokenizer, processor

    elif model_name == "mplug_owl2":
        from mplug_owl2.model.builder import load_pretrained_model

        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path, None, model_name, load_8bit=False, load_4bit=False, device="cuda"
        )
        model = model.cuda()

        return model, tokenizer, image_processor

    else:
        raise ValueError(f"Unsupported model name: {model_name}")
