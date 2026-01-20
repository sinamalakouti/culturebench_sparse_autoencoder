from pathlib import Path
import os
import random
import pandas as pd  # type: ignore
from pandas.core.nanops import na_accum_func
import yaml

benchmark_version = "v2"
culturebench_root = Path("/u/sem238/datasets/culturebench/")
culturebench_root = culturebench_root / benchmark_version
google_image_root = Path(
    # "/Users/sinamalakouti/Desktop/cultural_data/culturebench/groundtruth/culturebench_google/"
    "/u/sem238/datasets/culturebench/groundtruth/culturebench_googledddddd"
)
descriptor_paths = (
    Path(culturebench_root)
    / "descriptors"
    / "social_activities"
    / "llm_descriptors"
    # / "activity_setting_{descriptor_mode}.json"
)

output_dir = "../outputs"

# Module-level config - loaded once
_config = None


def load_config(config_path: str):
    """Load config once at startup."""
    global _config
    with open(config_path, "r") as f:
        _config = yaml.safe_load(f)


def get_llm_descriptors_path(descriptor_mode):
    """Get LLM descriptors path - reads from config if available, otherwise uses original logic."""
    # If config is loaded and has LLM descriptor path, use it
    if _config and "PATHS" in _config and "LLM_DESCRIPTORS_PATH" in _config["PATHS"]:
        return Path(_config["PATHS"]["LLM_DESCRIPTORS_PATH"])

    # Original logic (backward compatibility)
    if benchmark_version == "v1":
        assert (
            descriptor_mode == "base" or descriptor_mode == "country"
        ), f"Invalid descriptor mode: {descriptor_mode}"
        p = (
            Path(culturebench_root)
            / "descriptors"
            / "social_activities"
            / "llm_descriptors"
            / f"activity_descriptors_{descriptor_mode}.json"
        )
    elif benchmark_version == "v2":
        assert (
            descriptor_mode == "base" or descriptor_mode == "country"
        ), f"Invalid descriptor mode: {descriptor_mode}"
        p = (
            Path(culturebench_root)
            / "descriptors"
            / "llm_descriptors"
            / f"descriptors_{descriptor_mode}.json"
        )
    else:
        raise ValueError(f"Invalid benchmark version: {benchmark_version}")

    p = Path(
        "/afs/cs.pitt.edu/usr0/sem238/projects/CultureGen/data/llm_descriptors_proposer/gpt-4o/refiner/proposers_gemini-2.5-flash_gpt-4o/all_descriptors.json"
    )
    return p


def get_image_path(img_path):
    if os.path.exists(img_path):
        return img_path
    elif os.path.exists(img_path + ".png"):
        return img_path + ".png"
    elif os.path.exists(img_path + ".jpg"):
        return img_path + ".jpg"
    elif os.path.exists(img_path + ".jpeg"):
        return img_path + ".jpeg"
    else:
        img_path + ".png"


def read_google_image_mapping_csv(
    filter_by_clip_score=False, filter_by_selected_images=False
):
    if not filter_by_clip_score and not filter_by_selected_images:
        return pd.read_csv(os.path.join(google_image_root, "url_mapping_filtered.csv"))
    elif filter_by_selected_images:
        return pd.read_csv(os.path.join(google_image_root, "url_mapping_filtered.csv"))
    elif filter_by_clip_score:
        return pd.read_csv(os.path.join(google_image_root, "url_mapping_filtered.csv"))
    return None


def country_to_country_code(country):
    return country.upper()


def get_img_dir_name(activity, subactivity, country=None):
    activity = activity.replace(" ", "_")
    subactivity = subactivity.replace(" ", "_")
    country = country.replace(" ", "_")

    if country:
        return f"{activity}_{subactivity}_{country}"
    else:
        return f"{activity}_{subactivity}"


def get_cultural_bench_dir(T2I, gen_mode, desc_mode):
    T2I = get_T2I_name(T2I)
    dir_path = culturebench_root
    dir_path.mkdir(parents=True, exist_ok=True)
    dir_path = Path(dir_path)
    if "descriptor" in gen_mode:
        dir_path = dir_path / T2I / f"GEN_MODE_{gen_mode}_DESC_MODE_{desc_mode}"
    else:
        dir_path = dir_path / T2I / f"GEN_MODE_{gen_mode}"
    return dir_path


def get_cultural_bench_image_dir(
    activity, subactivity, country, T2I, gen_mode, desc_mode
):
    dir_path = get_cultural_bench_dir(T2I, gen_mode, desc_mode)
    dir_path = Path(dir_path) / "images"
    dir_path.mkdir(parents=True, exist_ok=True)

    activity = activity.replace(" ", "_")
    subactivity = subactivity.replace(" ", "_")
    country = country.replace(" ", "_")

    if "country" in gen_mode:
        dir_path = dir_path / country / activity / subactivity
    else:
        dir_path = dir_path / activity / subactivity

    return dir_path


def get_T2I_name(T2I):
    T2I = T2I.lower()
    if T2I == "dall-e-3":
        return "dall-e-3"
    elif T2I == "sd3.5" or T2I == "stable-diffusion-3.5-medium":
        return "stable-diffusion-3.5-medium"
    elif T2I == "flux1" or T2I == "black-forest-labs/flux.1-dev" or T2I == "flux.1-dev":
        return "FLUX.1-dev"
    elif T2I == "qwen-image":
        return "qwen-image"
    elif T2I == "gt_images":
        return "gt_images"
    elif T2I == "nano-banana" or T2I == "gemini-2.5-flash-image-preview":
        return "gemini-2.5-flash-image-preview"
    elif T2I == "gpt-image-1":
        return "gpt-image-1"
    else:
        raise ValueError(f"Invalid T2I: {T2I}")


def parse_image_number(image_path):
    image_name = str(image_path.stem)
    return int(image_name.split("_")[-1])


def get_image_paths(
    activity, subactivity, country, T2I, gen_mode, desc_mode, max_num_images=None
):
    dir_path = get_cultural_bench_image_dir(
        activity,
        subactivity,
        country_to_country_code(country),
        get_T2I_name(T2I),
        gen_mode,
        desc_mode,
    )

    # Get all PNG files, sort them by name, and take only the first 3
    all_images = sorted(list(set(dir_path.glob("*.png"))))
    if max_num_images:
        random.shuffle(all_images)
        return all_images[:max_num_images]
    else:
        return all_images


def get_culturebench_mllm_descriptor_path(
    gen_mode: str,
    llm_gen_desc_mode: str,
    T2I: str,
    mllm_desc_model: str,
    country: str = None,
) -> Path:
    mllm_desc_model = f"{mllm_desc_model}_desc_generator"
    T2I = get_T2I_name(T2I)

    if "descriptor" in gen_mode:
        if benchmark_version == "v1":
            parent_dir = (
                Path(culturebench_root)
                / T2I
                / f"GEN_MODE_{gen_mode}_DESC_MODE_{llm_gen_desc_mode}"
            )
        elif benchmark_version == "v2":
            parent_dir = (
                Path(culturebench_root)
                / T2I
                / f"GEN_MODE_{gen_mode}_DESC_MODE_{llm_gen_desc_mode}"
            )
    else:
        if benchmark_version == "v1":
            parent_dir = Path(culturebench_root) / T2I / f"GEN_MODE_{gen_mode}"
        elif benchmark_version == "v2":
            parent_dir = Path(culturebench_root) / T2I / f"GEN_MODE_{gen_mode}"

    mllm_desc_model = (
        mllm_desc_model if benchmark_version == "v1" else mllm_desc_model + "_axis"
    )
    if country:
        descriptor_path = (
            Path(parent_dir)
            / "evaluation"
            / "descriptors"
            / f"{mllm_desc_model}"
            / f"{country}"
            / f"descriptors_EVAL_MODE_base_base.json"
        )
    else:
        descriptor_path = (
            Path(parent_dir)
            / "evaluation"
            / "descriptors"
            / f"{mllm_desc_model}"
            / f"descriptors_EVAL_MODE_base_base.json"
        )
    return descriptor_path


# from pathlib import Path
# import os
# import random
# import pandas as pd  # type: ignore

# benchmark_version = "v2"
# culturebench_root = Path("/u/sem238/datasets/culturebench/")
# culturebench_root = culturebench_root / benchmark_version
# google_image_root = Path(
#     # "/Users/sinamalakouti/Desktop/cultural_data/culturebench/groundtruth/culturebench_google/"
#     "/u/sem238/datasets/culturebench/groundtruth/culturebench_googledddddd"
# )
# descriptor_paths = (
#     Path(culturebench_root)
#     / "descriptors"
#     / "social_activities"
#     / "llm_descriptors"
#     # / "activity_setting_{descriptor_mode}.json"
# )

# output_dir = "../outputs"

# """
# gen_mode: base, countryInPrompt, descriptive

# """


# def get_image_path(img_path):

#     if os.path.exists(img_path):
#         return img_path
#     elif os.path.exists(img_path + ".png"):
#         return img_path + ".png"
#     elif os.path.exists(img_path + ".jpg"):
#         return img_path + ".jpg"
#     elif os.path.exists(img_path + ".jpeg"):
#         return img_path + ".jpeg"
#     else:
#         img_path + ".png"
#         # raise ValueError(f"Image path {img_path} does not exist")


# def read_google_image_mapping_csv(
#     filter_by_clip_score=False, filter_by_selected_images=False
# ):
#     if not filter_by_clip_score and not filter_by_selected_images:
#         return pd.read_csv(os.path.join(google_image_root, "url_mapping_filtered.csv"))
#     elif filter_by_selected_images:
#         return pd.read_csv(os.path.join(google_image_root, "url_mapping_filtered.csv"))
#     elif filter_by_clip_score:
#         return pd.read_csv(os.path.join(google_image_root, "url_mapping_filtered.csv"))
#     return None


# def country_to_country_code(country):
#     return country.upper()


# def get_img_dir_name(activity, subactivity, country=None):
#     activity = activity.replace(" ", "_")
#     subactivity = subactivity.replace(" ", "_")
#     country = country.replace(" ", "_")

#     if country:
#         return f"{activity}_{subactivity}_{country}"
#     else:
#         return f"{activity}_{subactivity}"


# def get_cultural_bench_dir(T2I, gen_mode, desc_mode):
#     T2I = get_T2I_name(T2I)
#     dir_path = culturebench_root
#     dir_path.mkdir(parents=True, exist_ok=True)
#     dir_path = Path(dir_path)
#     if "descriptor" in gen_mode:
#         dir_path = dir_path / T2I / f"GEN_MODE_{gen_mode}_DESC_MODE_{desc_mode}"
#     else:
#         dir_path = dir_path / T2I / f"GEN_MODE_{gen_mode}"
#     return dir_path


# # def _get_image_dir_path(cultural_bench_dir, activity, subactivity, country):
# #     dir_path = cultural_bench_dir / "images"
# #     dir_path.mkdir(parents=True, exist_ok=True)
# #     return dir_path


# def get_cultural_bench_image_dir(
#     activity, subactivity, country, T2I, gen_mode, desc_mode
# ):
#     dir_path = get_cultural_bench_dir(T2I, gen_mode, desc_mode)
#     dir_path = Path(dir_path) / "images"
#     dir_path.mkdir(parents=True, exist_ok=True)

#     activity = activity.replace(" ", "_")
#     subactivity = subactivity.replace(" ", "_")
#     country = country.replace(" ", "_")

#     if "country" in gen_mode:
#         dir_path = (
#             dir_path
#             / country
#             / activity
#             / subactivity
#             # / get_img_dir_name(activity, subactivity, country)
#         )
#     else:
#         dir_path = (
#             dir_path
#             / activity
#             / subactivity
#             # / get_img_dir_name(activity, subactivity)
#         )

#     return dir_path


# def get_T2I_name(T2I):
#     T2I = T2I.lower()
#     if T2I == "dall-e-3":
#         return "dall-e-3"
#     elif T2I == "sd3.5" or T2I == "stable-diffusion-3.5-medium":
#         return "stable-diffusion-3.5-medium"
#     elif T2I == "flux1" or T2I == "black-forest-labs/flux.1-dev" or T2I == "flux.1-dev":
#         return "FLUX.1-dev"
#     elif T2I == "qwen-image":
#         return "qwen-image"
#     elif T2I == "gt_images":
#         return "gt_images"
#     else:
#         raise ValueError(f"Invalid T2I: {T2I}")


# def get_image_paths(
#     activity, subactivity, country, T2I, gen_mode, desc_mode, max_num_images=None
# ):
#     dir_path = get_cultural_bench_image_dir(
#         activity,
#         subactivity,
#         country_to_country_code(country),
#         get_T2I_name(T2I),
#         gen_mode,
#         desc_mode,
#     )

#     # Get all PNG files, sort them by name, and take only the first 3
#     all_images = sorted(list(set(dir_path.glob("*.png"))))
#     if max_num_images:
#         random.shuffle(all_images)
#         return all_images[:max_num_images]
#     else:
#         return all_images


# def get_llm_descriptors_path(descriptor_mode):
#     if benchmark_version == "v1":
#         assert (
#             descriptor_mode == "base" or descriptor_mode == "country"
#         ), f"Invalid descriptor mode: {descriptor_mode}"
#         p = (
#             Path(culturebench_root)
#             / "descriptors"
#             / "social_activities"
#             / "llm_descriptors"
#             / f"activity_descriptors_{descriptor_mode}.json"
#         )
#     elif benchmark_version == "v2":
#         assert (
#             descriptor_mode == "base" or descriptor_mode == "country"
#         ), f"Invalid descriptor mode: {descriptor_mode}"
#         p = (
#             Path(culturebench_root)
#             / "descriptors"
#             / "llm_descriptors"
#             / f"descriptors_{descriptor_mode}.json"
#         )
#     else:
#         raise ValueError(f"Invalid benchmark version: {benchmark_version}")
#     return p
#     # / "activity_setting_{descriptor_mode}.json"


# # def get_mllm_descriptor_path(
# #     parent_dir, descriptor_model_name, img_gen_mode, descriptor_mode
# # ):
# #     descriptor_path = (
# #         Path(parent_dir)
# #         / "descriptors"
# #         / f"{descriptor_model_name}"
# #         / f"descriptors_{img_gen_mode}_{descriptor_mode}.json"
# #     )
# #     descriptor_path.parent.mkdir(parents=True, exist_ok=True)
# #     return descriptor_path


# def get_culturebench_mllm_descriptor_path(
#     gen_mode: str,
#     llm_gen_desc_mode: str,
#     T2I: str,
#     mllm_desc_model: str,
#     country: str = None,
# ) -> Path:
#     mllm_desc_model = f"{mllm_desc_model}_desc_generator"
#     T2I = get_T2I_name(T2I)

#     if "descriptor" in gen_mode:
#         if benchmark_version == "v1":
#             parent_dir = (
#                 Path(culturebench_root)
#                 / T2I
#                 / f"GEN_MODE_{gen_mode}_DESC_MODE_{llm_gen_desc_mode}"
#             )
#         elif benchmark_version == "v2":
#             parent_dir = (
#                 Path(culturebench_root)
#                 / T2I
#                 / f"GEN_MODE_{gen_mode}_DESC_MODE_{llm_gen_desc_mode}"
#             )
#     else:
#         if benchmark_version == "v1":
#             parent_dir = Path(culturebench_root) / T2I / f"GEN_MODE_{gen_mode}"
#         elif benchmark_version == "v2":
#             parent_dir = Path(culturebench_root) / T2I / f"GEN_MODE_{gen_mode}"

#     mllm_desc_model = (
#         mllm_desc_model if benchmark_version == "v1" else mllm_desc_model + "_axis"
#     )
#     if country:
#         descriptor_path = (
#             Path(parent_dir)
#             / "evaluation"
#             / "descriptors"
#             / f"{mllm_desc_model}"
#             / f"{country}"
#             / f"descriptors_EVAL_MODE_base_base.json"
#         )
#     else:
#         descriptor_path = (
#             Path(parent_dir)
#             / "evaluation"
#             / "descriptors"
#             / f"{mllm_desc_model}"
#             / f"descriptors_EVAL_MODE_base_base.json"
#         )
#     # descriptor_path.parent.mkdir(parents=True, exist_ok=True)
#     return descriptor_path
