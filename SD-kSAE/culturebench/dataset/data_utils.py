import logging
import os
import json
from pathlib import Path
from typing import Dict

from tqdm.auto import tqdm as progress_bar
from culturegen.dataset.culturebench import (
    ACTIVITY_PROMPTS,
    COUNTRIES,
    ACTIVITY_TO_SUBACTIVITIES,
    PREFIX_PROMPT,
)
from culturegen.dataset.paths import (
    benchmark_version,
    get_culturebench_mllm_descriptor_path,
    get_image_path,
    get_image_paths,
    get_llm_descriptors_path,
    read_google_image_mapping_csv,
    google_image_root,
)
from culturegen.dataset.image_info import ImageItem, ImageItemCollection


# from tools.activity_descriptor_generator import (
#     generate_culturebench_llm_descriptors,
# )


def get_activities(culturebench_data: None):
    if benchmark_version == "v1":
        return list(ACTIVITY_TO_SUBACTIVITIES.keys())
    elif benchmark_version == "v2":
        return list(get_activities_from_json(culturebench_data))
    else:
        assert False, f"Invalid benchmark version: {benchmark_version}"


def get_subactivities(
    activity: str,
    country: str = None,
    culturebench_data: Dict = None,
    include_parent_activity=True,
):
    if benchmark_version == "v1":
        return ACTIVITY_TO_SUBACTIVITIES[activity]
    elif benchmark_version == "v2":
        return get_subactivities_from_json(
            activity, country, culturebench_data, include_parent_activity
        )
    else:
        assert False, f"Invalid benchmark version: {benchmark_version}"


def get_activity_subactivity_pairs():
    if benchmark_version == "v1":
        return list(ACTIVITY_PROMPTS.keys())
    elif benchmark_version == "v2":
        raise NotImplementedError(
            " get_activity_subactivity_pairs Not implemented for v2"
        )
    else:
        assert False, f"Invalid benchmark version: {benchmark_version}"


def get_countries(culturebench_data=None):
    if benchmark_version == "v1":
        return COUNTRIES
    elif benchmark_version == "v2":
        return get_countries_from_json(culturebench_data)
    else:
        assert False, f"Invalid benchmark version: {benchmark_version}"


def _get_base_prompt(activity, subactivity, country=None, add_prefix=True):

    if benchmark_version == "v1":
        assert (
            f"{activity}_{subactivity}" in ACTIVITY_PROMPTS
        ), f"key {activity}_{subactivity} not found in ACTIVITY_PROMPTS"
        if add_prefix:
            return PREFIX_PROMPT + ACTIVITY_PROMPTS[f"{activity}_{subactivity}"]
        else:
            return ACTIVITY_PROMPTS[f"{activity}_{subactivity}"]
    elif benchmark_version == "v2":
        _, prompts_data = load_culturebench_json()
        base_prompt = prompts_data[country][activity][subactivity]
        if add_prefix:
            return PREFIX_PROMPT + base_prompt
        else:
            return base_prompt


def _get_country_in_prompt(activity, subactivity, country, add_prefix=True):
    assert country in get_countries(), f"Country {country} not found in COUNTRIES"

    base_prompt = _get_base_prompt(activity, subactivity, country, add_prefix)
    return base_prompt + f" in {country_to_country_prompt(country)}"


def _get_descriptive_prompt(base_prompt, descriptors):
    if benchmark_version == "v1":
        assert descriptors is not None, "Descriptors are not loaded"
        prompt = base_prompt + f", featuring {', '.join(descriptors)}"
        return prompt
    elif benchmark_version == "v2":
        prompt = base_prompt + f", featuring {', '.join(descriptors)}"
        return prompt


def country_to_country_prompt(country):

    return country.replace("_", " ").lower().capitalize()


def get_descriptors(
    activity, subactivity, descriptor_mode, country=None, descriptors_path=None
):
    descriptors = None
    if "country" in descriptor_mode:
        assert (
            country is not None
        ), "Country is required for country-specific descriptors"

    if descriptors_path is None:
        descriptors_path = get_llm_descriptors_path(descriptor_mode)
        if not descriptors_path.exists():

            print(
                f"descriptor for {activity}_{subactivity}_{country} not found. Generating descriptors..."
            )
            # generate and save descriptors. path should be in the culture bench
            # generate_culturebench_llm_descriptors(descriptor_mode)

    assert descriptors_path is not None, "Descriptors path is not set"
    assert (
        descriptors_path.exists()
    ), f"Descriptors path {descriptors_path} does not exist"
    descriptors = json.load(open(descriptors_path))

    if "base" in descriptor_mode:
        if isinstance(descriptors[f"{activity}_{subactivity}"], dict):
            descriptors = descriptors[f"{activity}_{subactivity}"]["general"]
        else:
            descriptors = descriptors[f"{activity}_{subactivity}"]
    elif "country" in descriptor_mode:
        descriptors = descriptors[f"{activity}_{subactivity}"][country]
    if type(descriptors) == str:
        descriptors = descriptors.split("featuring")[1].split(",")
        descriptors = [d.strip() for d in descriptors if d.strip()]

    return descriptors


def get_descriptors_v2(
    activity, subactivity, descriptor_mode="base", country=None, descriptors_path=None
):
    descriptors = None
    if "country" in descriptor_mode:
        assert (
            country is not None
        ), "Country is required for country-specific descriptors"

    if descriptors_path is None:
        descriptors_path = get_llm_descriptors_path(descriptor_mode)
        logging.info(
            f"\n*************\ndescriptors_path: {descriptors_path}\n*************\n"
        )
        if not descriptors_path.exists():

            print(
                f"descriptor for {activity}_{subactivity}_{country} not found. Generating descriptors..."
            )
            # generate and save descriptors. path should be in the culture bench
            # generate_culturebench_llm_descriptors(descriptor_mode)

    assert descriptors_path is not None, "Descriptors path is not set"
    assert (
        descriptors_path.exists()
    ), f"Descriptors path {descriptors_path} does not exist"
    all_descriptors = json.load(open(descriptors_path))

    if "base" in descriptor_mode:
        assert False, "Not implemented"
    elif "country" in descriptor_mode:
        descriptors = all_descriptors[country][activity][subactivity]
        descriptors_text = {}
        for key in descriptors.keys():
            if key == "people":
                continue
            descriptors_text[key] = [
                descriptor["token"] for descriptor in descriptors[key]
            ]
        descriptors = descriptors_text
    if type(descriptors) == str:
        descriptors = descriptors.split("featuring")[1].split(",")
        descriptors = [d.strip() for d in descriptors if d.strip()]

    return descriptors


def get_prompt(
    activity,
    subactivity,
    gen_mode=None,
    desc_mode=None,
    country=None,
    descriptors=None,
    add_prefix=True,
):
    prompt = None
    if "base" in gen_mode:
        prompt = _get_base_prompt(activity, subactivity, country, add_prefix)
    elif "country" in gen_mode:
        prompt = _get_country_in_prompt(activity, subactivity, country, add_prefix)
    if "descriptor" in gen_mode:
        if descriptors is None:
            if benchmark_version == "v1":
                descriptors = get_descriptors(activity, subactivity, desc_mode, country)
            elif benchmark_version == "v2":
                descriptors = get_descriptors_v2(
                    activity, subactivity, desc_mode, country
                )
            else:
                assert False, f"Invalid benchmark version: {benchmark_version}"
        prompt = _get_descriptive_prompt(prompt, descriptors)
    assert prompt is not None, f"Invalid mode: {gen_mode}"
    return prompt


def get_prompt_from_json(
    activity: str, subactivity: str, country: str, prompts_data: dict
) -> str:
    """Get prompt from the JSON data and format it appropriately"""
    base_prompt = prompts_data[country][activity][subactivity]
    return base_prompt.lower()


def get_google_image_infos(
    filter_by_clip_score=False,
    filter_by_selected_images=False,
) -> ImageItemCollection:
    """
    Process the CSV file containing Google image information and create ImageItems.

    Args:
        filter_by_clip_score (bool): If True, select top 5 images by clip_score_country
        filter_by_selected_images (bool): If True, only use images with selected=1

    Returns:
        ImageItemCollection: Collection of ImageItems with unique URLs
    """
    # Read the CSV file
    df = read_google_image_mapping_csv(
        filter_by_clip_score=filter_by_clip_score,
        filter_by_selected_images=filter_by_selected_images,
    )

    # If filtering by selected images is enabled, filter the dataframe
    if filter_by_selected_images:
        df = df[df["selected"] == 1]

    # Remove duplicate URLs, keeping the first occurrence
    df = df.drop_duplicates(subset=["url"], keep="first")

    # If filtering by clip score is enabled, keep only top 5 images per group
    if filter_by_clip_score:
        # Group by activity, subactivity, country and get top 5 by clip_score_country
        df = (
            df.sort_values("clip_score_country", ascending=False)
            .groupby(["activity", "subactivity", "country"])
            .head(5)
        )

    # Create ImageItems
    mllm_descriptor_path = (
        Path(google_image_root) / "descriptors" / "gt_descriptors.json"
    )
    all_mllm_descriptors = load_mllm_descriptors(mllm_descriptor_path)
    items = []
    for _, row in df.iterrows():
        # Check if image exists
        image_path = get_image_path(
            os.path.join(str(google_image_root), row["image_path"])
        )
        if image_path is not None and os.path.exists(image_path):
            image_id_mllm_descriptor = f"{row['activity']}_{row['subactivity']}_{row['country']}_{Path(image_path).stem}"
            mllm_item = all_mllm_descriptors[image_id_mllm_descriptor]
            assert (
                image_id_mllm_descriptor in all_mllm_descriptors
            ), f"image_id_mllm_descriptor {image_id_mllm_descriptor} not found in all_mllm_descriptors for GT data"
            assert str(image_path) == str(
                mllm_item["path"]
            ), f"img_path {image_path} does not match mllm_item {mllm_item}"
            item_descriptors = mllm_item["descriptors"]
            assert (
                item_descriptors is not None
            ), f"item_descriptors is None for {image_id_mllm_descriptor} in GT data"

            item = ImageItem(
                path=image_path,
                activity=row["activity"],
                subactivity=row["subactivity"],
                country=row["country"],
                mllm_descriptors=item_descriptors,
                # Add additional attributes
                query_type=row["prompt_type"],  # rename prompt_type to query_type
                url=row["url"],
                clip_score_base=float(row["clip_score_base"]),
                clip_score_country=float(row["clip_score_country"]),
            )
            items.append(item)
        else:
            logging.warning(f"Image path {image_path} does not exist")
            print(
                f"Image path {image_path + '.png'} and {image_path + '.jpg'} do not exist"
            )

    # Create and return collection
    collection = ImageItemCollection(items)
    collection.summarize()  # Add summary to show distribution of images
    return collection


def save_image_collection(collection: ImageItemCollection, output_path: str) -> None:
    """Save the image collection to a JSON file."""
    collection.save_json(output_path)


def load_image_collection(json_path: str) -> ImageItemCollection:
    """Load the image collection from a JSON file."""
    return ImageItemCollection.load_json(json_path)


def load_culturebench_json(
    data_json_path: str = "data/culturebench.json",
    prompts_json_path: str = "data/culturebench_prompts.json",
) -> Dict:
    """
    Load culturebench data from JSON file.

    Args:
        json_path: Path to the culturebench.json file

    Returns:
        Dict containing the culturebench data structure
    """
    with open(data_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    with open(prompts_json_path, "r", encoding="utf-8") as f:
        prompts_data = json.load(f)
    return data, prompts_data


def get_activities_from_json(culturebench_data: Dict = None, country=None) -> list:
    """
    Get list of activities from JSON data.

    Args:
        culturebench_data: Loaded culturebench data, if None will load from file

    Returns:
        List of activity names
    """
    if culturebench_data is None:
        culturebench_data, _ = load_culturebench_json()

    # Get all unique activities from all countries
    activities = set()
    if country:
        activities = culturebench_data[country].keys()
    else:
        for country_data in culturebench_data.values():
            activities.update(country_data.keys())

    return sorted(list(activities))


def get_subactivities_from_json(
    activity: str,
    country: str = None,
    culturebench_data: Dict = None,
    include_parent_activity=True,
) -> list:
    """
    Get list of subactivities for a given activity from JSON data.

    Args:
        activity: Activity name
        culturebench_data: Loaded culturebench data, if None will load from file

    Returns:
        List of subactivity names for the given activity
    """
    if culturebench_data is None:
        culturebench_data, _ = load_culturebench_json()

    if country:
        subactivities = culturebench_data[country][activity]
        subactivities = set(subactivities)
        if include_parent_activity:
            subactivities.add(activity)
        return sorted(list(subactivities))

    # Get all subactivities for this activity across all countries
    subactivities = set()
    for country_data in culturebench_data.values():
        if activity in country_data:
            subactivities.update(country_data[activity])

    if include_parent_activity:
        subactivities.add(activity)

    return sorted(list(subactivities))


def get_countries_from_json(culturebench_data: Dict = None) -> list:
    """
    Get list of countries from JSON data.

    Args:
        culturebench_data: Loaded culturebench data, if None will load from file

    Returns:
        List of country names
    """
    if culturebench_data is None:
        culturebench_data, _ = load_culturebench_json()

    return sorted(list(culturebench_data.keys()))


def get_descriptors_by_axis(mllm_descriptors, axis_name=None):
    """
    Get descriptors from MLLM descriptor data, handling both legacy and 5-axis formats.

    Args:
        mllm_descriptors: Descriptor data (can be list for legacy or dict for 5-axis)
        axis_name: Specific axis to get descriptors from (for 5-axis format)
                  Options: "setting", "objects", "attire", "interaction", "spatial"
                  If None, returns all descriptors as a flat list

    Returns:
        List of descriptors for the specified axis or all descriptors combined
    """
    if mllm_descriptors is None:
        return []

    if isinstance(mllm_descriptors, list):
        # Legacy format - return all descriptors
        return mllm_descriptors
    elif isinstance(mllm_descriptors, dict):
        # 5-axis format
        if axis_name is None:
            # Return all descriptors as a flat list
            all_descriptors = []
            for axis_descriptors in mllm_descriptors.values():
                if isinstance(axis_descriptors, list):
                    all_descriptors.extend(axis_descriptors)
            return all_descriptors
        else:
            # Return descriptors for specific axis
            return mllm_descriptors.get(axis_name, [])
    else:
        logging.warning(f"Unexpected descriptor format: {type(mllm_descriptors)}")
        return []


def get_all_axis_names():
    """Get list of all available axis names for 5-axis descriptors."""
    return ["setting", "objects", "attire", "interaction", "spatial"]


def convert_legacy_to_axis_format(legacy_descriptors):
    """
    Convert legacy descriptor format (list) to 5-axis format (dict).
    This is useful for backward compatibility.

    Args:
        legacy_descriptors: List of descriptors in legacy format

    Returns:
        Dictionary with all descriptors placed in the "objects" axis
    """
    if not isinstance(legacy_descriptors, list):
        return {}

    return {
        "setting": [],
        "objects": legacy_descriptors,
        "attire": [],
        "interaction": [],
        "spatial": [],
    }


def convert_axis_to_legacy_format(axis_descriptors):
    """
    Convert 5-axis descriptor format (dict) to legacy format (list).
    This flattens all axis descriptors into a single list.

    Args:
        axis_descriptors: Dictionary of descriptors organized by axis

    Returns:
        List of all descriptors combined
    """
    if not isinstance(axis_descriptors, dict):
        return []

    all_descriptors = []
    for axis_descriptors_list in axis_descriptors.values():
        if isinstance(axis_descriptors_list, list):
            all_descriptors.extend(axis_descriptors_list)

    return all_descriptors


def get_culturebench_image_infos(
    gen_mode,
    llm_gen_desc_mode,
    T2I,
    activities=None,
    subactivities=None,
    countries=None,
    mllm_desc_model=None,
    eval_desc_mode=None,
    include_mllm_descriptors=False,
) -> ImageItemCollection:
    activities = activities or get_activities()
    countries = countries or get_countries()

    if include_mllm_descriptors:
        mllm_descriptor_path = get_culturebench_mllm_descriptor_path(
            gen_mode, llm_gen_desc_mode, T2I, mllm_desc_model, country=None
        )
        all_mllm_descriptors = load_mllm_descriptors(mllm_descriptor_path)
        # print("******ALL MLLM DESCRIPTORS******")
        # print(all_mllm_descriptors)
    else:
        all_mllm_descriptors = None

    image_infos = []
    for activity in activities:
        subactivities = get_subactivities(activity)
        for subactivity in subactivities:
            if "country" in gen_mode or "country" in llm_gen_desc_mode:
                # Only iterate over countries if either mode needs country-specific handling
                for country in countries:
                    img_paths = get_image_paths(
                        activity, subactivity, country, T2I, gen_mode, llm_gen_desc_mode
                    )

                    for img_path in img_paths:
                        # Get descriptors for different modes
                        # llm_gt_descriptors = get_descriptors(
                        #     activity, subactivity, "llm_gt", country
                        # )
                        llm_gen_descriptors = get_descriptors(
                            activity, subactivity, llm_gen_desc_mode, country
                        )
                        # mllm_descriptors = get_descriptors(
                        #     activity, subactivity, "mllm", country
                        # )
                        if include_mllm_descriptors:
                            image_id_mllm_descriptor = f"{activity}_{subactivity}_{country}_{Path(img_path).stem}"
                            # print(
                            #     "IMAGE ID MLLM DESCRIPTOR : ", image_id_mllm_descriptor
                            # )
                            mllm_item = all_mllm_descriptors[image_id_mllm_descriptor]
                            assert (
                                image_id_mllm_descriptor in all_mllm_descriptors
                            ), f"image_id_mllm_descriptor {image_id_mllm_descriptor} not found in all_mllm_descriptors"
                            assert str(img_path) == str(
                                mllm_item["path"]
                            ), f"img_path {img_path} does not match mllm_item {mllm_item}"
                            item_descriptors = mllm_item["descriptors"]
                        else:
                            item_descriptors = None

                        image_infos.append(
                            ImageItem(
                                path=img_path,
                                activity=activity,
                                subactivity=subactivity,
                                country=(
                                    country if "country" in gen_mode else "general"
                                ),
                                llm_gt_desc_mode=None,
                                llm_gen_desc_mode=llm_gen_desc_mode,
                                gen_mode=gen_mode,
                                mllm_descp_mode=None,
                                llm_gt_descriptors=None,
                                llm_gen_descriptors=llm_gen_descriptors,
                                mllm_descriptors=item_descriptors,
                            )
                        )
            else:
                # For base modes, don't iterate over countries
                country = "general"
                img_paths = get_image_paths(
                    activity, subactivity, country, T2I, gen_mode, llm_gen_desc_mode
                )

                for img_path in img_paths:
                    # # Get descriptors for different modes
                    # llm_gt_descriptors = get_descriptors(
                    #     activity, subactivity, "llm_gt", None
                    # )
                    llm_gen_descriptors = get_descriptors(
                        activity, subactivity, llm_gen_desc_mode, country
                    )
                    # mllm_descriptors = get_descriptors(
                    #     activity, subactivity, "mllm", None
                    # )

                    if include_mllm_descriptors:
                        image_id_mllm_descriptor = (
                            f"{activity}_{subactivity}_{country}_{Path(img_path).stem}"
                        )
                        # print("IMAGE ID MLLM DESCRIPTOR : ", image_id_mllm_descriptor)
                        mllm_item = all_mllm_descriptors[image_id_mllm_descriptor]
                        # print("mllm_item : ", mllm_item)
                        assert (
                            image_id_mllm_descriptor in all_mllm_descriptors
                        ), f"image_id_mllm_descriptor {image_id_mllm_descriptor} not found in all_mllm_descriptors"
                        assert str(img_path) == str(
                            mllm_item["path"]
                        ), f"img_path {str(img_path)} does not match mllm_item {mllm_item['path']}"
                        item_descriptors = mllm_item["descriptors"]
                    else:
                        item_descriptors = None

                    image_infos.append(
                        ImageItem(
                            path=img_path,
                            activity=activity,
                            subactivity=subactivity,
                            country=country,
                            llm_gt_desc_mode=None,
                            llm_gen_desc_mode=llm_gen_desc_mode,
                            gen_mode=gen_mode,
                            mllm_descp_mode=None,
                            llm_gt_descriptors=None,
                            llm_gen_descriptors=llm_gen_descriptors,
                            mllm_descriptors=item_descriptors,
                        )
                    )

    # Create a collection and summarize it
    collection = ImageItemCollection(image_infos)
    collection.summarize()
    return collection


def get_culturebench_image_infos_v2(
    gen_mode,
    llm_gen_desc_mode,
    T2I,
    activities=None,
    subactivities=None,
    countries=None,
    mllm_desc_model=None,
    include_mllm_descriptors=False,
    json_path: str = "data/culturebench.json",
) -> ImageItemCollection:
    """
    Version 2 of get_culturebench_image_infos that reads data from JSON file instead of hardcoded constants.

    Args:
        gen_mode: Generation mode
        llm_gen_desc_mode: LLM generation descriptor mode
        T2I: Text-to-image model identifier
        activities: List of activities to process, if None uses all from JSON
        subactivities: List of subactivities to process, if None uses all from JSON
        countries: List of countries to process, if None uses all from JSON
        mllm_desc_model: MLLM descriptor model
        eval_desc_mode: Evaluation descriptor mode
        include_mllm_descriptors: Whether to include MLLM descriptors
        json_path: Path to the culturebench.json file

    Returns:
        ImageItemCollection: Collection of ImageItems
    """
    # Load culturebench data from JSON
    culturebench_data, _ = load_culturebench_json(json_path)

    print("json_path : ", json_path)

    # Get activities, subactivities, and countries from JSON if not provided
    activities = activities or get_activities(culturebench_data)
    countries = countries or get_countries(culturebench_data)

    if include_mllm_descriptors:
        mllm_descriptor_path = get_culturebench_mllm_descriptor_path(
            gen_mode, llm_gen_desc_mode, T2I, mllm_desc_model, country=None
        )
        all_mllm_descriptors = load_mllm_descriptors(mllm_descriptor_path)
    else:
        all_mllm_descriptors = None

    image_infos = []
    pbar = progress_bar(desc="Loading image items")  # single bar, no fixed total
    for activity in activities:
        # Get subactivities for this activity from JSON
        activity_subactivities = get_subactivities(
            activity, country=None, culturebench_data=culturebench_data
        )

        # Filter by provided subactivities if specified
        # if subactivities is not None:
        #     activity_subactivities = [
        #         s for s in activity_subactivities if s in subactivities
        #     ]

        for subactivity in activity_subactivities:
            if "country" in gen_mode or "country" in llm_gen_desc_mode:
                # Only iterate over countries if either mode needs country-specific handling
                for country in countries:
                    # Check if this country has this activity-subactivity combination
                    if (
                        country in culturebench_data
                        and activity in culturebench_data[country]
                        and (
                            subactivity in culturebench_data[country][activity]
                            or subactivity == activity
                        )
                    ):
                        # print(
                        #     f"Processing {activity} {subactivity} {country} {gen_mode} {llm_gen_desc_mode}"
                        # )

                        img_paths = get_image_paths(
                            activity,
                            subactivity,
                            country,
                            T2I,
                            gen_mode,
                            llm_gen_desc_mode,
                        )

                        if len(img_paths) == 0:
                            print(
                                f"No image paths found for {activity} {subactivity} {country} {gen_mode} {llm_gen_desc_mode}"
                            )

                        for img_path in img_paths:
                            # llm_gen_descriptors = get_descriptors(
                            #     activity, subactivity, llm_gen_desc_mode, country
                            # )

                            if include_mllm_descriptors:
                                image_id_mllm_descriptor = f"{activity}_{subactivity}_{country}_{Path(img_path).stem}"
                                assert (
                                    image_id_mllm_descriptor in all_mllm_descriptors
                                ), f"image_id_mllm_descriptor {image_id_mllm_descriptor} not found in all_mllm_descriptors"

                                mllm_item = all_mllm_descriptors[
                                    image_id_mllm_descriptor
                                ]
                                assert str(img_path) == str(
                                    mllm_item["path"]
                                ), f"img_path {img_path} does not match mllm_item {mllm_item}"

                                item_descriptors = mllm_item["descriptors"]
                            else:
                                item_descriptors = None

                            image_infos.append(
                                ImageItem(
                                    path=img_path,
                                    activity=activity,
                                    subactivity=subactivity,
                                    country=(
                                        country if "country" in gen_mode else "general"
                                    ),
                                    llm_gt_desc_mode=None,
                                    llm_gen_desc_mode=llm_gen_desc_mode,
                                    gen_mode=gen_mode,
                                    mllm_descp_mode=None,
                                    llm_gt_descriptors=None,
                                    llm_gen_descriptors=None,
                                    mllm_descriptors=item_descriptors,
                                )
                            )
                            pbar.update(1)
            else:
                # For base modes, don't iterate over countries
                assert False, "v2 is not supported for base modes"
                country = "general"
                img_paths = get_image_paths(
                    activity, subactivity, country, T2I, gen_mode, llm_gen_desc_mode
                )

                for img_path in img_paths:
                    # llm_gen_descriptors = get_descriptors(
                    #     activity, subactivity, llm_gen_desc_mode, country
                    # )

                    if include_mllm_descriptors:
                        image_id_mllm_descriptor = (
                            f"{activity}_{subactivity}_{country}_{Path(img_path).stem}"
                        )
                        mllm_item = all_mllm_descriptors[image_id_mllm_descriptor]
                        assert (
                            image_id_mllm_descriptor in all_mllm_descriptors
                        ), f"image_id_mllm_descriptor {image_id_mllm_descriptor} not found in all_mllm_descriptors"
                        assert str(img_path) == str(
                            mllm_item["path"]
                        ), f"img_path {str(img_path)} does not match mllm_item {mllm_item['path']}"
                        item_descriptors = mllm_item["descriptors"]
                    else:
                        item_descriptors = None

                    image_infos.append(
                        ImageItem(
                            path=img_path,
                            activity=activity,
                            subactivity=subactivity,
                            country=country,
                            llm_gt_desc_mode=None,
                            llm_gen_desc_mode=llm_gen_desc_mode,
                            gen_mode=gen_mode,
                            mllm_descp_mode=None,
                            llm_gt_descriptors=None,
                            llm_gen_descriptors=None,
                            mllm_descriptors=item_descriptors,
                        )
                    )

    # Create a collection and summarize it
    collection = ImageItemCollection(image_infos)
    # collection.summarize()
    pbar.close()
    return collection


def load_mllm_descriptors(mllm_descriptor_path: str) -> Dict:
    """
    Load MLLM descriptors from a JSON file generated by InternVLDescriptorGenerator.

    Args:
        mllm_descriptor_path: Path to descriptor JSON file

    Returns:
        Dict mapping image_id to a dict containing all info including descriptors

    Note:
        The descriptors can be generated using two methods:
        1. Legacy templates: Single flat list of descriptors
        2. 5-axis templates: Dictionary with descriptors organized by axis
           {
             "setting": ["descriptor1", "descriptor2"],
             "objects": ["descriptor3", "descriptor4"],
             "attire": ["descriptor5", "descriptor6"],
             "interaction": ["descriptor7", "descriptor8"],
             "spatial": ["descriptor9", "descriptor10"]
           }

        The template_type field in the JSON indicates which method was used.
    """
    mllm_descriptor_path = Path(mllm_descriptor_path)
    if not mllm_descriptor_path.suffix.lower() == ".json":
        raise ValueError(f"Expected .json file, got: {mllm_descriptor_path.suffix}")

    with open(mllm_descriptor_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # Convert to standardized format and filter failed generations
    mllm_data = {}
    failed_count = 0
    total_count = len(raw_data)

    for image_id, info in raw_data.items():
        # Skip failed generations
        if not info.get("success", True):
            failed_count += 1
            # continue

        img_path = info.get("path") or info.get("original_path")
        img_path = get_image_path(img_path)

        # Handle different descriptor formats
        descriptors = info["descriptors"]
        template_type = info.get("template_type", "legacy")

        # Validate descriptor structure based on template type
        if template_type == "5-axis":
            # For 5-axis templates, descriptors should be a dictionary
            assert isinstance(
                descriptors, dict
            ), f"Expected dict for 5-axis descriptors, got {type(descriptors)} for {image_id}"
        else:
            # For legacy templates, descriptors should be a list
            assert isinstance(
                descriptors, list
            ), f"Expected list for legacy descriptors, got {type(descriptors)} for {image_id}"

        mllm_data[image_id] = {
            "activity": info["activity"],
            "subactivity": info["subactivity"],
            "country": info["country"],
            "descriptors": descriptors,
            "path": img_path,
            "template_type": template_type,
        }

    if failed_count > 0:
        logging.warning(
            f"Skipped {failed_count}/{total_count} entries due to failed generation or missing descriptors"
        )

    if not mllm_data:
        raise ValueError(f"No valid descriptor data found in {mllm_descriptor_path}")

    return mllm_data
