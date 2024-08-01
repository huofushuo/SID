import argparse
import os
import random
import json
from PIL import Image
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import tqdm
from datetime import datetime

from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from torchvision.utils import save_image

from pope_loader import POPEDataSet
from minigpt4.common.dist_utils import get_rank
from minigpt4.models import load_preprocess

from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *

from eval_utils.shr.shr_utils import *
from eval_utils.shr.gpt_utils import *
from vcd_add_noise import add_diffusion_noise
from vcd_sample import evolve_vcd_sampling
import warnings
warnings.filterwarnings("ignore")
time = datetime.now().strftime('%m-%d-%H:%M')
print(time)
evolve_vcd_sampling()
# python pope_eval.py --model minigpt4 --data_path /home/hfs/e/llm/mscoco/val2014 --pope-type random --gpu-id 0 --beam 5 --scale_factor 50 --threshold 15 --num_attn_candidates 5 --penalty_weights 1
# python pope_eval.py --model llava-1.5 --data_path /home/hfs/e/llm/mscoco/val2014 --pope-type random --gpu-id 0 --beam 5 --scale_factor 50 --threshold 15 --num_attn_candidates 5 --penalty_weights 1
# python pope_eval.py --model shikra --data_path /home/hfs/e/llm/mscoco/val2014 --pope-type random --gpu-id 0 --beam 5 --scale_factor 50 --threshold 15 --num_attn_candidates 5 --penalty_weights 1


MODEL_EVAL_CONFIG_PATH = {
    "minigpt4": "eval_configs/minigpt4_eval.yaml",
    "instructblip": "eval_configs/instructblip_eval.yaml",
    "lrv_instruct": "eval_configs/lrv_instruct_eval.yaml",
    "shikra": "eval_configs/shikra_eval.yaml",
    "llava-1.5": "eval_configs/llava-1.5_eval.yaml",
}

POPE_PATH = {
    "coco_random": "pope/coco/coco_pope_random.json",
    "coco_popular": "pope/coco/coco_pope_popular.json",
    "coco_adversarial": "pope/coco/coco_pope_adversarial.json",
    "gpa_random": "pope/gpa/gqa_pope_seem_random.json",
    "gpa_popular": "pope/gpa/gqa_pope_seem_popular.json",
    "gpa_adversarial": "pope/gpa/gqa_pope_seem_adversarial.json",
    "aokvqa_random": "pope/aokvqa/aokvqa_pope_seem_random.json",
    "aokvqa_popular": "pope/aokvqa/aokvqa_pope_seem_popular.json",
    "aokvqa_adversarial": "pope/aokvqa/aokvqa_pope_seem_adversarial.json",
}

INSTRUCTION_TEMPLATE = {
    "minigpt4": "###Human: <Img><ImageHere></Img> <question> ###Assistant:",
    "instructblip": "<ImageHere><question>",
    "lrv_instruct": "###Human: <Img><ImageHere></Img> <question> ###Assistant:",
    "shikra": "USER: <im_start><ImageHere><im_end> <question> ASSISTANT:",
    "llava-1.5": "USER: <ImageHere> <question> ASSISTANT:"
}


def parse_args():
    parser = argparse.ArgumentParser(description="POPE-Adv evaluation on LVLMs.")
    parser.add_argument("--model", type=str, default="llava-1.5", help="model")
    parser.add_argument("--pope-type", type=str, default="coco_random", help="model")
    # parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    # parser.add_argument("--data-path", type=str, default="/home/hfs/e/llm/GQA/raw/images/", help="data path")
    # parser.add_argument("--data-path", type=str, default="/home/hfs/e/llm/mscoco/val2014", help="data path")
    # parser.add_argument("--batch-size", type=int, default=1, help="batch size")
    # parser.add_argument("--num_workers", type=int, default=1, help="num workers")
    # parser.add_argument("--answers-file", type=str, default="/home/hfs/llm/OPERA-main/log/llava-1.5/pope/")

    parser.add_argument("--use-fast-v", action='store_true', default=False)
    parser.add_argument("--fast-v-inplace", default=False)
    parser.add_argument("--fast-v-attention-rank", type=int, default=16)
    parser.add_argument("--fast-v-attention-rank-add", type=int, default=100)
    parser.add_argument("--fast-v-agg-layer", type=int, default=10)
    parser.add_argument('--fast-v-sys-length', default=None, type=int, help='the length of system prompt')
    parser.add_argument('--fast-v-image-token-length', default=None, type=int, help='the length of image token')
    # opera-beamsearch
    parser.add_argument("--beam", type=int, default=1)
    parser.add_argument("--sample", action='store_true', default=True)
    parser.add_argument("--scale_factor", type=float, default=50)
    parser.add_argument("--threshold", type=int, default=15)
    parser.add_argument("--num_attn_candidates", type=int, default=5)
    parser.add_argument("--penalty_weights", type=float, default=1.0)
    parser.add_argument("--opera", default=False)
    # vision contrastive decoding
    parser.add_argument("--noise_step", type=int, default=500)
    parser.add_argument("--use-cd", action='store_true', default=False)
    parser.add_argument("--use-icd", action='store_true', default=False)
    parser.add_argument("--use-vcd", action='store_true', default=False)
    parser.add_argument("--cd-alpha", type=float, default=1)
    parser.add_argument("--cd-beta", type=float, default=0.1)
    # SHR parameters
    parser.add_argument("--api-key", type=str, default='', help="key to the OPENAI API.")
    parser.add_argument("--vg-path", type=str, default='/home/hfs/e/llm/Visual_Genome_Dataset_V1_dot_2/raw/data/', help="path to vg file.")
    parser.add_argument("--shr-path", type=str, default='/home/hfs/llm/OPERA-main/eval_utils/shr', help="path to SHR annotation file.")
    parser.add_argument("--no-gpt-judge", default=False, action='store_true', help="whether not to do GPT evaluation. If True, only evaluate ngram repitition.")
    args = parser.parse_args()
    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


def main():

    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    args.cfg_path = MODEL_EVAL_CONFIG_PATH[args.model]
    args.pope_path = POPE_PATH[args.pope_type]
    cfg = Config(args)

    setup_seeds(cfg)
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    # setup openai
    setup_openai(args.api_key)

    # ========================================
    #             Model Initialization
    # ========================================
    print('Initializing Model')

    model_config = cfg.model_cfg
    model_config.device_8bit = args.gpu_id
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to(device)
    model.eval()

    # set model decoding config
    if args.model == "instructblip":
        if args.use_fast_v == True:
            model.llm_model.config.use_fast_v = args.use_fast_v
            model.llm_model.config.fast_v_inplace = args.fast_v_inplace
            model.llm_model.config.fast_v_sys_length = args.fast_v_sys_length
            model.llm_model.config.fast_v_image_token_length = args.fast_v_image_token_length
            model.llm_model.config.fast_v_attention_rank = args.fast_v_attention_rank
            model.llm_model.config.fast_v_attention_rank_add = args.fast_v_attention_rank_add
            model.llm_model.config.fast_v_agg_layer = args.fast_v_agg_layer
        else:
            model.llm_model.config.use_fast_v = args.use_fast_v
        model.llm_model.model.reset_fastv()
    else:
        if args.use_fast_v == True:
            model.llama_model.config.use_fast_v = args.use_fast_v
            model.llama_model.config.fast_v_inplace = args.fast_v_inplace
            model.llama_model.config.fast_v_sys_length = args.fast_v_sys_length
            model.llama_model.config.fast_v_image_token_length = args.fast_v_image_token_length
            model.llama_model.config.fast_v_attention_rank = args.fast_v_attention_rank
            model.llama_model.config.fast_v_attention_rank_add = args.fast_v_attention_rank_add
            model.llama_model.config.fast_v_agg_layer = args.fast_v_agg_layer
        else:
            model.llama_model.config.use_fast_v = args.use_fast_v
        model.llama_model.model.reset_fastv()

    vis_processors, txt_processors = load_preprocess(cfg.get_config().preprocess)
    # vis_processors.do_normalize = False
    print(vis_processors["eval"].transform)
    print("Done!")

    # visual genome annotations
    val_images = json.load(open(os.path.join(args.shr_path, "val_images_final.json")))
    vg_image_data = json.load(open(os.path.join(args.vg_path, "image_data.json")))
    id2path = {
        _data["image_id"]:os.path.join(args.vg_path, _data["url"].split("/")[-2], _data["url"].split("/")[-1]) 
        for _data in vg_image_data
    }
    id2img = {_data["image_id"]:_data for _data in vg_image_data}
    region = json.load(open(os.path.join(args.vg_path, "region_descriptions.json")))
    id2reg = {r["regions"][0]["image_id"]:r for r in region}
    
    judgement = {}
    run_all = ['run1']
    for run in run_all:
        judgement[run] = {}
    _gram1, _gram2, _gram3, _gram4 = 0, 0, 0, 0
    
    # factual information
    factual_inf = {}
    factual_part1 = os.path.join(args.shr_path, "shr_factual_part1.jsonl")
    factual_part2 = os.path.join(args.shr_path, "shr_factual_part2.jsonl")
    for line in open(factual_part1).readlines():
        factual = json.loads(line)
        image_id, factuals = list(factual.keys())[0], list(factual.values())[0]
        factual_inf[image_id] = factuals
    for line in open(factual_part2).readlines():
        factual = json.loads(line)
        image_id, factuals = list(factual.keys())[0], list(factual.values())[0]
        factual_inf[image_id] = factuals

    for _data in tqdm.tqdm(val_images):
        image_id = _data["image_id"]
        image_path = id2path[int(image_id)]
        image = Image.open(image_path).convert("RGB")
        # Similar operation in model_worker.py
        image_tensor = vis_processors["eval"](image)
        if type(image_tensor) is list:
            image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
        else:
            image_tensor = image_tensor.to(model.device, dtype=torch.float16)

        inp = "Describe this image in detail."
        template = INSTRUCTION_TEMPLATE[args.model]
        qu = template.replace("<question>", inp)
        image = image_tensor.to(device).unsqueeze(0)

        if args.use_cd:
            image_cd = image.to(device)
        elif args.use_vcd:
            image_cd = add_diffusion_noise(image, args.noise_step)
            image_cd = image.to(device)
        else:
            image_cd = None

        if args.use_icd:
            text_cd = 'You are a confused object detector.'
            if args.model == 'shikra':
                prompt_cd = qu[0].split("<im_end>")[0] + "<im_end>" + ' ' + text_cd + qu[0].split("<im_end>")[-1]
            elif args.model == 'llava-1.5' or args.model == 'instructblip':
                prompt_cd = qu[0].split("<ImageHere>")[0] + "<ImageHere>" + ' ' + text_cd + qu[0].split("<ImageHere>")[-1]
            # elif args.model == 'lrv_instruct' or args.model == 'minigpt4':
            else:
                prompt_cd = qu[0].split("</Img>")[0] + "</Img>" + ' ' + text_cd + qu[0].split("</Img>")[-1]
        else:
            text_cd = None

        with torch.inference_mode():
            with torch.no_grad():
                outputs = model.generate(
                    # {"image": norm(image), "prompt":qu},
                    prompt = qu,
                    image = image.half(),
                    images_cd=(image_cd.half() if image_cd is not None else None),
                    prompt_cd =(prompt_cd if text_cd is not None else None),
                    use_nucleus_sampling=args.sample, 
                    num_beams=args.beam,
                    max_new_tokens=512,
                    output_attentions=True,
                    opera_decoding=args.opera,
                    scale_factor=args.scale_factor,
                    threshold=args.threshold,
                    num_attn_candidates=args.num_attn_candidates,
                    penalty_weights=args.penalty_weights,
                    use_cache=True,
                    # do_sample=True,
                )[0]

        # get GPT judgement
        description = get_desc(id2img, id2reg, int(image_id))
        model_cap_sep, is_repeated = get_model_cap(outputs)
        # calculate repetition
        gram1 = cal_repetition(outputs,1)
        gram2 = cal_repetition(outputs,2)
        gram3 = cal_repetition(outputs,3)
        gram4 = cal_repetition(outputs,4)
        _gram1 += gram1
        _gram2 += gram2
        _gram3 += gram3
        _gram4 += gram4
            
        # skip gpt judgement 
        if args.no_gpt_judge:
            continue
            
        factual_text = ""
        if str(image_id) in factual_inf:
            for text in factual_inf[str(image_id)]:
                factual_text += text
                factual_text += "\n"
        # GPT judgement
        judge_prompt = GPT_JUDGE_PROMPT.format(description, factual_text, model_cap_sep)
        if len(judge_prompt) > 15000:
            print(f"skip {image_id} for too long prompt!")
            continue
        
        
        for run in run_all:
            while True:
                judge = get_gpt_response(prompt=judge_prompt)
                if "Judgement" not in judge:
                    print(f"No judgement found for {image_id}")
                    continue
                else:
                    break
            # post-process
            final_judge = post_process_no_revise(judge, outputs)
            judgement[run][image_id] = {
                "raw_judgement": judge,
                "model_response": outputs,
                "judgement": final_judge,
            }
        
    if args.no_gpt_judge:
        print(f"gram-1 repetition: {round(_gram1/len(val_images), 3)}")
        print(f"gram-2 repetition: {round(_gram2/len(val_images), 3)}")
        print(f"gram-3 repetition: {round(_gram3/len(val_images), 3)}")
        print(f"gram-4 repetition: {round(_gram4/len(val_images), 3)}")
    else:
        base_eval_path = "./results/shr/{}".format(args.model)
        localtime = time.asctime( time.localtime(time.time()) ).replace(' ', '_')
        if not os.path.exists(os.path.join(base_eval_path)):
            os.mkdir(os.path.join(base_eval_path))
        # dump config file
        eval_path = os.path.join(os.path.join(base_eval_path, localtime))
        os.mkdir(eval_path)
        # save metrics
        metrics = {}
        for run in run_all:
            metrics[run] = {}
            get_metric(judgement[run], metrics[run])
        # repetition
        metrics['gram-1-repetition'] = round(_gram1/len(val_images), 3)
        metrics['gram-2-repetition'] = round(_gram2/len(val_images), 3)
        metrics['gram-3-repetition'] = round(_gram3/len(val_images), 3)
        metrics['gram-4-repetition'] = round(_gram4/len(val_images), 3)
        # halucination ratio
        metrics["mean_hal_ratio"] = round(
            sum(metrics[run]["hal_sents_ratio"] for run in run_all)/len(run_all), 3
        )
        metrics["model_base"] = args.model_base
        metrics["model_path"] = args.model_path
        # dump judgement file
        with open(os.path.join(base_eval_path, localtime, 'judgement.json'), "w") as f:
            json.dump(judgement, f)
        # dump metric file
        with open(os.path.join(base_eval_path, localtime, 'metrics.json'), "w") as f:
            json.dump(metrics, f)









if __name__ == "__main__":
    main()
