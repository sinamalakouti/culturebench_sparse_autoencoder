from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from collections import defaultdict
import logging
import json
from datetime import datetime
from pathlib import Path


@dataclass
class ImageItem:
    path: str
    activity: str
    subactivity: str
    country: str
    llm_gt_desc_mode: str = None
    llm_gen_desc_mode: str = None
    gen_mode: str = None
    mllm_descp_mode: str = None
    llm_gt_descriptors: List[str] = None
    llm_gen_descriptors: List[str] = None
    mllm_descriptors: List[str] = None
    _additional_attrs: Dict[str, Any] = field(default_factory=dict)

    def __init__(self, **kwargs):
        """
        Initialize ImageItem with dynamic keyword arguments.
        Required fields: path, activity, subactivity, country
        Any additional fields will be stored in _additional_attrs
        """
        # First, ensure required fields are present
        required_fields = {"path", "activity", "subactivity", "country"}
        missing_fields = required_fields - set(kwargs.keys())
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")

        # Initialize _additional_attrs
        self._additional_attrs = {}

        # Process all kwargs
        for key, value in kwargs.items():
            if key in self.__annotations__ and key != "_additional_attrs":
                # If it's a defined field, set it directly
                setattr(self, key, value)
            else:
                # If it's not a defined field, store it in _additional_attrs
                self._additional_attrs[key] = value

    def __getattr__(self, name: str) -> Any:
        """Allow accessing additional attributes."""
        if name in self._additional_attrs:
            return self._additional_attrs[name]
        raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        """Allow setting additional attributes."""
        if name in self.__annotations__ or name == "_additional_attrs":
            super().__setattr__(name, value)
        else:
            self._additional_attrs[name] = value

    def to_dict(self) -> Dict:
        """Convert ImageItem to dictionary including additional attributes."""
        # Get all standard dataclass fields
        base_dict = {
            key: getattr(self, key)
            for key in self.__annotations__
            if key != "_additional_attrs"
        }
        # Add additional attributes
        base_dict.update(self._additional_attrs)
        return base_dict

    @classmethod
    def from_dict(cls, data: Dict) -> "ImageItem":
        """Create ImageItem from dictionary, handling additional attributes."""
        # Create instance with all attributes
        return cls(**data)

    def matches(self, **kwargs) -> bool:
        """Check if this item matches all provided criteria, including additional attributes."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                if getattr(self, key) != value:
                    return False
            elif key in self._additional_attrs:
                if self._additional_attrs[key] != value:
                    return False
            else:
                return False
        return True


class ImageItemCollection:
    def __init__(self, items: Optional[List[ImageItem]] = None):
        self.items = items or []
        self._activity_index: Dict[str, List[ImageItem]] = defaultdict(list)
        self._country_index: Dict[str, List[ImageItem]] = defaultdict(list)
        self._subactivity_index: Dict[str, List[ImageItem]] = defaultdict(list)

        # Build indices for faster searching
        for item in self.items:
            self._add_to_indices(item)

    def __len__(self):
        return len(self.items)

    def _add_to_indices(self, item: ImageItem) -> None:
        """Add an item to all search indices."""
        self._activity_index[item.activity].append(item)
        self._country_index[item.country].append(item)
        self._subactivity_index[item.subactivity].append(item)

    def add_item(self, item: ImageItem) -> None:
        """Add a single item to the collection."""
        self.items.append(item)
        self._add_to_indices(item)

    def find_by_activity(self, activity: str) -> List[ImageItem]:
        """Find all items with the given activity."""
        return list(self._activity_index[activity])

    def find_by_country(self, country: str) -> List[ImageItem]:
        """Find all items from the given country."""
        return list(self._country_index[country])

    def find_by_subactivity(self, subactivity: str) -> List[ImageItem]:
        """Find all items with the given subactivity."""
        return list(self._subactivity_index[subactivity])

    def find_by_criteria(self, **kwargs) -> List[ImageItem]:
        """
        Find all items matching the given criteria.
        Example: find_by_criteria(activity='wedding', country='India')
        """
        # Start with the smallest index if possible
        if "activity" in kwargs:
            items = self._activity_index[kwargs["activity"]]
        elif "country" in kwargs:
            items = self._country_index[kwargs["country"]]
        elif "subactivity" in kwargs:
            items = self._subactivity_index[kwargs["subactivity"]]
        else:
            items = self.items

        return [item for item in items if item.matches(**kwargs)]

    def group_by(self, *keys: str) -> Dict[tuple, List[ImageItem]]:
        """
        Group items by specified attribute keys.

        Args:
            *keys: Variable number of attribute names to group by (e.g., 'activity', 'subactivity', 'country')

        Returns:
            Dict[tuple, List[ImageItem]]: Dictionary with tuples of key values as keys and lists of matching ImageItems as values
        """
        if not keys:
            logging.warning("No keys provided for grouping")
            return {}
        if not self.items:
            logging.warning("No items in the collection")
            return {}

        grouped = defaultdict(list)
        for item in self.items:
            key = tuple(getattr(item, k) for k in keys)

            grouped[key].append(item)
        return grouped

    @classmethod
    def from_info_list(cls, info_list: List[dict]) -> "ImageItemCollection":
        """Create an ImageItemCollection from a list of image info dictionaries."""
        items = [
            ImageItem(
                path=info["path"],
                activity=info["activity"],
                subactivity=info["subactivity"],
                country=info["country"],
                llm_gt_desc_mode=info.get("llm_gt_desc_mode", ""),
                llm_gen_desc_mode=info.get("llm_gen_desc_mode", ""),
                gen_mode=info.get("gen_mode", ""),
                mllm_descp_mode=info.get("mllm_descp_mode", ""),
                llm_gt_descriptors=info.get("llm_gt_descriptors", []),
                llm_gen_descriptors=info.get("llm_gen_descriptors", []),
                mllm_descriptors=info.get("mllm_descriptors", []),
            )
            for info in info_list
        ]
        return cls(items)

    def summarize(self):
        """Summarize the collection with counts and distributions."""
        logging.info("=" * 50)
        logging.info("IMAGE INFO SUMMARY")
        logging.info("=" * 50)
        logging.info(f"Total number of images: {len(self.items)}")

        # Use our existing indices for counts
        activity_counts = {k: len(v) for k, v in self._activity_index.items()}
        country_counts = {k: len(v) for k, v in self._country_index.items()}
        subactivity_counts = {k: len(v) for k, v in self._subactivity_index.items()}

        # Log activity distribution
        logging.info("\n" + "-" * 30)
        logging.info("Activity distribution:")
        logging.info("-" * 30)
        for activity, count in activity_counts.items():
            logging.info(
                f"  {activity}: {count} images ({count/len(self.items)*100:.1f}%)"
            )

        # Log subactivity distribution
        logging.info("\n" + "-" * 30)
        logging.info("Subactivity distribution:")
        logging.info("-" * 30)
        for subactivity, count in subactivity_counts.items():
            logging.info(
                f"  {subactivity}: {count} images ({count/len(self.items)*100:.1f}%)"
            )

        # Log country distribution
        logging.info("\n" + "-" * 30)
        logging.info("Country distribution:")
        logging.info("-" * 30)
        for country, count in country_counts.items():
            logging.info(
                f"  {country}: {count} images ({count/len(self.items)*100:.1f}%)"
            )

        logging.info("=" * 50)

    def to_dict(self) -> Dict:
        """Convert collection to dictionary with metadata."""
        return {
            "metadata": {
                "total_items": len(self.items),
                "activities": list(self._activity_index.keys()),
                "countries": list(self._country_index.keys()),
                "subactivities": list(self._subactivity_index.keys()),
                "creation_date": datetime.now().isoformat(),
            },
            "items": [item.to_dict() for item in self.items],
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ImageItemCollection":
        """Create collection from dictionary."""
        items = [ImageItem.from_dict(item_data) for item_data in data["items"]]
        return cls(items)

    def save_json(self, path: str) -> None:
        """Save collection to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load_json(cls, path: str) -> "ImageItemCollection":
        """Load collection from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
