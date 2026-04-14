#create different functions which use the different translation models to translate the given source
# text into the target language text

#languages of interest are: kinyawarda, swahili, afan oromo, tigrinya and amharic
# Kinyarwanda rw
# Amharic am
# Afan Oromo om
# Tigrinya ti
# Swahili sw
import argparse
import csv 
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, T5Tokenizer, T5ForConditionalGeneration
import os
from transformers import pipeline


def write_to_file(output_path, translate_directory, target_language, output):
     #create a file in the output directory and then write the output there
    output_file_dir = os.path.join(output_path, translate_directory)
    os.makedirs(output_file_dir, exist_ok=True)
    output_file_path = os.path.join(output_file_dir, target_language+'.txt')

    with open(output_file_path, 'a', encoding='utf-8') as output_file:
        output_file.write(output.strip()+'\n')
    return

def get_llm_model_and_tokenizer(model_id):
    if "madlad" in model_id:
        tokenizer=T5Tokenizer.from_pretrained(model_id)
        model=T5ForConditionalGeneration.from_pretrained(
                model_id,
                device_map="auto",
                torch_dtype=torch.bfloat16
            )
    elif "translategemma" in model_id:
            model=pipeline(
                "text-generation",
                model=model_id,
                device_map="auto",
                torch_dtype=torch.bfloat16
            )
            tokenizer=None
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model=AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map="auto"
        )
    return model, tokenizer

#target_languages, source_data_path, source_language, source_text, output_data_path
def translate_with_gemma(tokenizer, 
                            model, 
                            target_language, 
                            source_path, 
                            source_language, 
                            source_text, 
                            output_path,
                            translation_directory):
    prompt=f"""<|turn>user
    You are a professional translator.
    Translate the following text from {source_language} to {target_language}.
    Output ONLY the translation, no explanations.

    Text: {source_text}<turn|>
    <|turn>model
    <|channel>thought
    <channel|>"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.3,
        do_sample=True
    )
    result=tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)

    write_to_file(output_path, translation_directory, target_language, result)

    return


def translate_with_afrique_gemma(tokenizer, 
                                    model, 
                                    target_language, 
                                    source_path, 
                                    source_language, 
                                    source_text, 
                                    output_path):
    
    prompt=f"""You are a professional translator.
    Translate the following text from {source_language} to {target_language}.
    Output ONLY the translation, no explanations.
    
    Text:{source_text}"""

    inputs=tokenizer(prompt, return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )
    output=tokenizer.decode(generated_ids[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)

    write_to_file(output_path, "afrique_gemma_4b_translations", target_language, output)

    return

def translate_with_afrique_qwen(tokenizer, 
                                    model, 
                                    target_language, 
                                    source_path, 
                                    source_language, 
                                    source_text, 
                                    output_path):
    prompt=f"""You are a professional translator.
    Translate the following text from {source_language} to {target_language}.
    Output ONLY the translation, no explanations.
    
    Text:{source_text}"""

    inputs=tokenizer(prompt, return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=100,
    )
    output=tokenizer.decode(generated_ids[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)

    write_to_file(output_path, "afrique_qwen_8b_translations", target_language, output)
    return


def translate_with_afrique_llama(tokenizer, 
                                    model, 
                                    target_language, 
                                    source_path, 
                                    source_language, 
                                    source_text, 
                                    output_path):
    prompt=f"""You are a professional translator.
    Translate the following text from {source_language} to {target_language}.
    Output ONLY the translation, no explanations.
    
    Text:{source_text}"""

    inputs=tokenizer(prompt, return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )
    output=tokenizer.decode(generated_ids[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)

    write_to_file(output_path, "afrique_llama_8b_translations", target_language, output)
    return


def translate_with_translate_gemma(gemma_model, 
                                    target_language, 
                                    source_path, 
                                    source_language, 
                                    source_text, 
                                    output_path,
                                    translation_directory):

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "source_lang_code": f"{source_language}",
                    "target_lang_code": f"{target_language}",
                    "text": f"{source_text}",
                }
            ],
        }
    ]

    returned_output = gemma_model(text_inputs=messages, max_new_tokens=200)
    output=returned_output[0]["generated_text"][-1]["content"]
    write_to_file(output_path, translation_directory, target_language, output)
    return


def translate_math_with_translate_gemma(gemma_model,
                                        target_language,
                                        source_path,
                                        source_language,
                                        source_text,
                                        output_path,
                                        translation_directory):
    """
    Math-specific translation using the prompt from AfriqueLLM Appendix A.2.
    Preserves LaTeX, formulas, and the <problem><think><eos> structure.
    max_new_tokens is set high enough to handle full reasoning traces.
    """
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "source_lang_code": f"{source_language}",
                    "target_lang_code": f"{target_language}",
                    "text": (
                        f"You are a {source_language}-to-{target_language} translator "
                        f"for mathematical content. Translate the provided math problem, "
                        f"reasoning, and answer while preserving:\n"
                        f"- All numbers, formulas, and formatting\n"
                        f"- Mathematical notation and markup\n"
                        f"- Named entities and tone\n\n"
                        f"Input structure:\n"
                        f"<problem>[Original Problem]</problem>\n"
                        f"<think>[Original Reasoning]</think>\n"
                        f"[Final Answer] <eos>\n\n"
                        f"Output structure:\n"
                        f"<problem>[Translated problem]</problem>\n"
                        f"<think>[Translated reasoning]</think>\n"
                        f"[Translated Final Answer] <eos>\n\n"
                        f"Ensure translations are fluent, coherent, and complete. "
                        f"Return only the translation without additional commentary.\n\n"
                        f"{source_text}"
                    ),
                }
            ],
        }
    ]

    returned_output = gemma_model(text_inputs=messages, max_new_tokens=4096)
    output = returned_output[0]["generated_text"][-1]["content"]
    write_to_file(output_path, translation_directory, target_language, output)
    return


def translate_web_with_translate_gemma(gemma_model,
                                       target_language,
                                       source_path,
                                       source_language,
                                       source_text,
                                       output_path,
                                       translation_directory):
    """
    Web domain translation using the prompt from AfriqueLLM Appendix A.2.
    Preserves formatting, inline markup, numerals, and named entities.
    """
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "source_lang_code": f"{source_language}",
                    "target_lang_code": f"{target_language}",
                    "text": (
                        f"You are a professional translator.\n"
                        f"Translate the user text from {source_language} into {target_language}.\n"
                        f"Preserve meaning, tone, formatting, inline markup, numerals, and "
                        f"named entities exactly.\n"
                        f"For long texts, ensure the translation is fluent, coherent and complete. "
                        f"Make sure to translate all parts of the text.\n"
                        f"Return only the translation without additional commentary.\n\n"
                        f"{source_text}"
                    ),
                }
            ],
        }
    ]

    returned_output = gemma_model(text_inputs=messages, max_new_tokens=4096)
    output = returned_output[0]["generated_text"][-1]["content"]
    write_to_file(output_path, translation_directory, target_language, output)
    return


def translate_with_madlad(model,
                            tokenizer,
                            target_language, 
                            source_path, 
                            source_language, 
                            source_text, 
                            output_path,
                            translation_directory):
    input_text = f"<2{target_language}> {source_text}"
    inputs=tokenizer(input_text, return_tensors="pt").to(model.device)
    returned_outputs=model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=False,
    )
    output=tokenizer.decode(returned_outputs[0], skip_special_tokens=True)
    write_to_file(output_path, translation_directory, target_language, output)
    return

def translate_with_tiny_aya(tokenizer, 
                                    model, 
                                    target_language, 
                                    source_path, 
                                    source_language, 
                                    source_text, 
                                    output_path,
                                    translation_directory):
    messages = [
                {"role": "user", "content": f"""You are a professional translator. Translate the following text from {source_language} to {target_language}.
                Output ONLY the translation, no explanations.
                Text:{source_text}"""},
                ]
    
    inputs = tokenizer.apply_chat_template(
	messages,
	add_generation_prompt=True,
	tokenize=True,
	return_dict=True,
	return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(**inputs, max_new_tokens=256)
    output=tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    write_to_file(output_path, translation_directory, target_language, output)
    return




if __name__ == '__main__':
    #get the source language parsed
    #parser.add_argument("--source_language", type=str, required=true, help="Provide the ISO code of the source language")
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_languages", type=str, nargs='+', required=True, help="Provide a list of the target languages")
    parser.add_argument("--source_data_path", type=str, required=True, help="Provide the path to where the source data is stored")
    parser.add_argument("--output_data_path", type=str, required=True, help="Provide the path to where the output data should be stored")
    parser.add_argument("--models", choices=["translate_gemma_4b", "translate_gemma_12b", "translate_gemma_27b", "afrique_gemma", "afrique_qwen", "afrique_llama", "gemma3_4b", "gemma3_27b", "gemma4_26b", "madlad", "tiny_aya"], required=True, help="Select which of the models to use for the translation task")

    #parse the arguments obtained from the command line
    args = parser.parse_args()
    translate_languages = args.target_languages
    source_data_path = args.source_data_path
    output_data_path = args.output_data_path
    model_choice = args.models

    #initialize the gemma 4 model
    gemma3_4b_model_id="google/gemma-3-4b-it"
    gemma3_27b_model_id="google/gemma-3-27b-it"
    gemma4_26b_model_id="google/gemma-4-26B-A4B-it"
    afrique_gemma_4b_model_id = "McGill-NLP/AfriqueGemma-4B"
    afrique_qwen_8b_model_id="McGill-NLP/AfriqueQwen-8B"
    afrique_llama_8b_model_id="McGill-NLP/AfriqueLlama-8B"
    translategemma_4b_model_id="google/translategemma-4b-it"
    translategemma_12b_model_id="google/translategemma-12b-it"
    translategemma_27b_model_id="google/translategemma-27b-it"
    madlad_model_id="google/madlad400-10b-mt"
    cohere_tiny_aya_model_id="CohereLabs/tiny-aya-global"

    if model_choice == "gemma3_4b":
        gemma3_4b_model, gemma3_4b_tokenizer = get_llm_model_and_tokenizer(gemma3_4b_model_id)
    if model_choice == "gemma3_27b":
        gemma3_27b_model, gemma3_27b_tokenizer = get_llm_model_and_tokenizer(gemma3_27b_model_id)
    if model_choice == "gemma4_26b":
        gemma4_26b_model, gemma4_26b_tokenizer = get_llm_model_and_tokenizer(gemma4_26b_model_id)
    if model_choice == "afrique_gemma":
        afrique_gemma_model, afrique_gemma_model_tokenizer = get_llm_model_and_tokenizer(afrique_gemma_4b_model_id)
    if model_choice == "afrique_qwen":
        afrique_qwen_model, afrique_qwen_model_tokenizer = get_llm_model_and_tokenizer(afrique_qwen_8b_model_id)
    if model_choice == "afrique_llama":
        afrique_llama_model, afrique_llama_model_tokenizer = get_llm_model_and_tokenizer(afrique_llama_8b_model_id)
    if model_choice == "translate_gemma_4b":
        translate_gemma_4b_model, translate_gemma_4b_model_tokenizer = get_llm_model_and_tokenizer(translategemma_4b_model_id)
    if model_choice == "translate_gemma_12b":
        translate_gemma_12b_model, translate_gemma_12b_model_tokenizer = get_llm_model_and_tokenizer(translategemma_12b_model_id)
    if model_choice == "translate_gemma_27b":
        translate_gemma_27b_model, translate_gemma_27b_model_tokenizer = get_llm_model_and_tokenizer(translategemma_27b_model_id)
    if model_choice == "tiny_aya":
        aya_model, aya_tokenizer = get_llm_model_and_tokenizer(cohere_tiny_aya_model_id)
    if model_choice == "madlad":
        madlad_model, madlad_tokenizer = get_llm_model_and_tokenizer(madlad_model_id)


    #open the source file
    with open(source_data_path, 'r', encoding='utf-8') as source_input:
        source_content = csv.reader(source_input)

        next(source_content)

        for source_content_line in source_content:
            source_language = source_content_line[1]
            source_text = source_content_line[2]

            #call the functions based on the arguments obtained
            if model_choice == "gemma3_4b":
                translation_directory= gemma3_4b_model_id.split('/')[-1]
                for translate_language in translate_languages:
                    translate_with_gemma(gemma3_4b_tokenizer, 
                                            gemma3_4b_model, 
                                            translate_language, 
                                            source_data_path, 
                                            source_language, 
                                            source_text, 
                                            output_data_path,
                                            translation_directory)
            
            if model_choice == "gemma3_27b":
                translation_directory= gemma3_27b_model_id.split('/')[-1]
                for translate_language in translate_languages:
                    translate_with_gemma(gemma3_27b_tokenizer, 
                                            gemma3_27b_model, 
                                            translate_language, 
                                            source_data_path, 
                                            source_language, 
                                            source_text, 
                                            output_data_path,
                                            translation_directory)
            
            if model_choice == "gemma4_26b":
                translation_directory= gemma4_26b_model_id.split('/')[-1]
                for translate_language in translate_languages:
                    translate_with_gemma(gemma4_26b_tokenizer, 
                                            gemma4_26b_model, 
                                            translate_language, 
                                            source_data_path, 
                                            source_language, 
                                            source_text, 
                                            output_data_path,
                                            translation_directory)
            if model_choice == "afrique_gemma":
                for translate_language in translate_languages:
                    translate_with_afrique_gemma(afrique_gemma_model_tokenizer,
                                                    afrique_gemma_model,
                                                    translate_language,
                                                    source_data_path,
                                                    source_language,
                                                    source_text,
                                                    output_data_path)

            if model_choice == "afrique_qwen":
                for translate_language in translate_languages:
                    translate_with_afrique_qwen(afrique_qwen_model_tokenizer,
                                                afrique_qwen_model,
                                                translate_language,
                                                source_data_path,
                                                source_language,
                                                source_text,
                                                output_data_path)
            
            if model_choice == "afrique_llama":
                for translate_language in translate_languages:
                    translate_with_afrique_llama(afrique_llama_model_tokenizer,
                                                afrique_llama_model,
                                                translate_language,
                                                source_data_path,
                                                source_language,
                                                source_text,
                                                output_data_path)

            if model_choice == "translate_gemma_4b":
                translation_directory= translategemma_4b_model_id.split('/')[-1]
                for translate_language in translate_languages:
                    translate_with_translate_gemma(translate_gemma_4b_model,
                                            translate_language, 
                                            source_data_path, 
                                            source_language, 
                                            source_text, 
                                            output_data_path,
                                            translation_directory)
            
            if model_choice == "translate_gemma_12b":
                translation_directory= translategemma_12b_model_id.split('/')[-1]
                for translate_language in translate_languages:
                    translate_with_translate_gemma(translate_gemma_12b_model,
                                            translate_language, 
                                            source_data_path, 
                                            source_language, 
                                            source_text, 
                                            output_data_path,
                                            translation_directory)
            
            if model_choice == "translate_gemma_27b":
                translation_directory= translategemma_27b_model_id.split('/')[-1]
                for translate_language in translate_languages:
                    translate_with_translate_gemma(translate_gemma_27b_model,
                                            translate_language, 
                                            source_data_path, 
                                            source_language, 
                                            source_text, 
                                            output_data_path,
                                            translation_directory)
            
            if model_choice == "madlad":
                translation_directory=madlad_model_id.split('/')[-1]
                for translate_language in translate_languages:
                    translate_with_madlad(madlad_model,
                                            madlad_tokenizer,
                                            translate_language,
                                            source_data_path,
                                            source_language,
                                            source_text,
                                            output_data_path,
                                            translation_directory)


            if model_choice == "tiny_aya":
                translation_directory=cohere_tiny_aya_model_id.split('/')[-1]
                for translate_language in translate_languages:
                    translate_with_tiny_aya(aya_tokenizer,
                                            aya_model,
                                            translate_language,
                                            source_data_path,
                                            source_language,
                                            source_text,
                                            output_data_path,
                                            translation_directory)
