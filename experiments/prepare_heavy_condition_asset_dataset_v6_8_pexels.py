#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import prepare_heavy_condition_asset_dataset_v5_32_pexels as v5


ROOT = Path(os.environ.get("H3_CONDITION_CACHE_ROOT", Path(__file__).resolve().parent))
OUT = ROOT / "artifacts/heavy_condition_asset_dataset_v6_8_organic_pexels_hotmatch_32aligned"


CASES = [
    {
        "id": "case00_flower_shop_bouquet_handoff",
        "theme": "flower arranging, bouquet wrapping, and shop display merged into one florist handoff",
        "semantic_merge": "flower texture, wrapping action, and storefront display combine into a coherent bouquet purchase story",
        "prompt": "Use Video 1 for the flower-arrangement texture, Video 2 for bouquet wrapping and hand contact, and Video 3 for the flower-shop display. Generate a bright florist-shop shot where a florist finishes a bouquet, wraps it neatly, and hands it to a customer beside a colorful counter, preserving petal detail, paper folds, and soft storefront light.",
        "refs": [
            ("pexels_6911353_flower_arrangement_table", "6911353", "detail/action: florist arranging flowers on a table"),
            ("pexels_10026929_bouquet_hands", "10026929", "subject/action: hands holding or wrapping a bouquet"),
            ("pexels_6933521_flower_shop_counter", "6933521", "environment: colorful flower-shop display or counter"),
        ],
    },
    {
        "id": "case01_rainy_bus_stop_commute",
        "theme": "rainy street, commuter umbrella movement, and bus-stop context merged into one commute scene",
        "semantic_merge": "rain texture, waiting commuter motion, and wet transit street context combine into a rainy bus-stop story",
        "prompt": "Use Video 1 for the rain-soaked city street, Video 2 for umbrella commuter motion, and Video 3 for the bus-stop or transit-side framing. Generate a rainy commute shot where a person waits under an umbrella near a bus stop as traffic passes on wet pavement, preserving rainfall streaks, reflections, and urban scale.",
        "refs": [
            ("pexels_15661481_rainy_city_bus_traffic", "15661481", "environment/weather: rainy city traffic with bus and wet street reflections"),
            ("pexels_5687574_umbrella_rain_commuter", "5687574", "subject/action: pedestrian or commuter moving under an umbrella"),
            ("pexels_36370582_wet_transit_waiting_area", "36370582", "context: wet sidewalk or transit waiting area"),
        ],
    },
    {
        "id": "case02_vineyard_harvest_wine_tasting",
        "theme": "vineyard rows, grape harvest detail, and wine pour merged into one farm-to-glass sequence",
        "semantic_merge": "vineyard environment, grape handling, and wine service detail combine into a harvest tasting story",
        "prompt": "Use Video 1 for the vineyard rows and daylight, Video 2 for grape cluster or harvest hand detail, and Video 3 for wine pouring or tasting motion. Generate a vineyard tasting shot where a worker gathers grapes in the rows and the scene resolves into a glass of wine being poured at a rustic outdoor table, preserving grape texture, leaf movement, and warm late-day light.",
        "refs": [
            ("pexels_5528415_vineyard_rows_daylight", "5528415", "environment: vineyard rows in daylight"),
            ("pexels_9947672_grape_cluster_harvest", "9947672", "detail/action: grapes on the vine or harvest handling"),
            ("pexels_8093235_wine_pour_tasting", "8093235", "detail/action: wine pouring or tasting glass"),
        ],
    },
    {
        "id": "case03_pizza_kitchen_oven_service",
        "theme": "pizza dough, kitchen preparation, and oven service merged into one restaurant cooking shot",
        "semantic_merge": "dough shaping, chef hand motion, and hot service context combine into a coherent pizza workflow",
        "prompt": "Use Video 1 for pizza dough shaping, Video 2 for chef preparation motion, and Video 3 for oven or serving context. Generate a pizzeria kitchen shot where dough is stretched, topped, moved toward a hot oven, and served on a wooden counter, preserving flour dust, cheese texture, and warm kitchen light.",
        "refs": [
            ("pexels_7172187_pizza_dough_shaping", "7172187", "action: pizza dough shaping on a work surface"),
            ("pexels_7172193_pizza_chef_preparation", "7172193", "subject/action: chef preparing pizza ingredients"),
            ("pexels_6603831_pizza_oven_service", "6603831", "environment/action: pizza oven or serving counter"),
        ],
    },
    {
        "id": "case04_fashion_atelier_sewing_fitting",
        "theme": "fabric sewing, machine detail, and fitting-room adjustment merged into one atelier shot",
        "semantic_merge": "textile texture, sewing-machine action, and garment fitting combine into one fashion workshop story",
        "prompt": "Use Video 1 for fabric handling, Video 2 for sewing-machine motion, and Video 3 for garment fitting or mannequin context. Generate a fashion atelier shot where a tailor guides fabric through a sewing machine, lifts the garment, and adjusts it on a model or mannequin, preserving thread detail, fabric folds, and studio lighting.",
        "refs": [
            ("pexels_8517568_woman_sewing_fabric", "8517568", "detail/action: woman sewing fabric with a machine"),
            ("pexels_32866095_denim_sewing_machine", "32866095", "action: sewing machine stitching denim fabric"),
            ("pexels_6460122_seamstress_measuring_tape", "6460122", "subject/context: seamstress, measuring tape, and tailoring workspace"),
        ],
    },
    {
        "id": "case05_seafood_market_harbor_counter",
        "theme": "fresh fish counter, market vendor motion, and harbor context merged into one seafood market scene",
        "semantic_merge": "fish detail, vendor handling, and coastal market context combine into a coherent morning market story",
        "prompt": "Use Video 1 for fresh fish and ice-counter texture, Video 2 for vendor handling or market interaction, and Video 3 for harbor or seafood-stall context. Generate a morning seafood-market shot where a vendor arranges fish on crushed ice and hands an order across the counter while harbor light and passing shoppers remain coherent in the background.",
        "refs": [
            ("pexels_8352337_red_fish_on_ice", "8352337", "detail: fresh fish on crushed ice"),
            ("pexels_992684_seafood_market_stall", "992684", "environment/action: seafood market stall with fresh fish"),
            ("pexels_38224900_fish_market_gathering", "38224900", "context: outdoor fish market crowd and baskets"),
        ],
    },
    {
        "id": "case06_art_gallery_painting_opening",
        "theme": "gallery walls, artist painting, and visitor motion merged into one opening-night scene",
        "semantic_merge": "gallery geometry, artwork creation, and visitor viewing combine into an art-opening story",
        "prompt": "Use Video 1 for gallery-wall and visitor spacing, Video 2 for artist brush or painting motion, and Video 3 for artwork close-up or exhibit lighting. Generate an art-gallery opening shot where visitors move past framed paintings while an artist adjusts a canvas near the wall, preserving clean gallery geometry, brush motion, and soft overhead light.",
        "refs": [
            ("pexels_7986429_museum_painting_discussion", "7986429", "environment/people: museum visitors discussing paintings"),
            ("pexels_8347848_artist_painting_canvas", "8347848", "subject/action: artist painting abstract art on canvas"),
            ("pexels_6607408_easel_painting_studio", "6607408", "detail/environment: easel, canvas, and studio gallery lighting"),
        ],
    },
    {
        "id": "case07_living_room_weather_tv",
        "theme": "living-room television, remote interaction, and rainy window weather merged into one indoor weather scene",
        "semantic_merge": "TV screen setup, remote-control action, and rainy window texture combine into one home weather-broadcast story",
        "prompt": "Use Video 1 for the living-room TV screen and couch layout, Video 2 for remote-control hand interaction, and Video 3 for rain on the window. Generate a cozy living-room shot where someone turns on a weather broadcast while rain streaks down the window behind the couch, preserving TV reflections, hand motion on the remote, and cool rainy daylight.",
        "refs": [
            ("pexels_5571655_living_room_tv_remote", "5571655", "environment/object: living-room TV screen and media console"),
            ("pexels_7981370_family_watching_tv", "7981370", "subject/action: people sitting on a sofa watching TV"),
            ("pexels_35180822_raindrops_window_night", "35180822", "weather/detail: raindrops on glass with night traffic lights"),
        ],
    },
]


def configure_v5_helpers() -> None:
    v5.ROOT = ROOT
    v5.OUT = OUT
    v5.CASES_DIR = OUT / "cases"
    v5.SHEETS_DIR = OUT / "contact_sheets"
    v5.OVERVIEWS_DIR = OUT / "overviews"
    for candidate in (
        Path(
            "/home/yitongl/code/streaming_h3/RAVEN/venv/lib/python3.10/site-packages/"
            "imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"
        ),
        Path(
            "/lustre/fsw/portfolios/nvr/users/yitongl/miniconda3/envs/h3-nrt/lib/python3.11/site-packages/"
            "imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"
        ),
        Path("/usr/bin/ffmpeg"),
    ):
        if candidate.exists():
            v5.FFMPEG = candidate
            break


def main() -> None:
    configure_v5_helpers()
    if OUT.exists():
        shutil.rmtree(OUT)
    v5.CASES_DIR.mkdir(parents=True)
    v5.SHEETS_DIR.mkdir(parents=True)
    v5.OVERVIEWS_DIR.mkdir(parents=True)

    records = []
    seen_ids: set[str] = set()
    for case in CASES:
        records.append(v5.build_new_case(case, len(records), seen_ids))

    manifest = {
        "name": "heavy_condition_asset_dataset_v6_8_organic_pexels_hotmatch_32aligned",
        "case_count": len(records),
        "refs_per_case": 3,
        "output_target": {"width": 1376, "height": 768, "fps": 24, "seconds": 5.0},
        "reference_policy": {
            "source": "Pexels videos surfaced through Pexels search pages and download endpoint",
            "seconds": 5.0,
            "fps": 24,
            "preserve_source_orientation": True,
            "pad_to_output_aspect": False,
            "target": "about 720p while keeping original aspect ratio; dimensions are 32-aligned for Ref2VA visual patching",
        },
        "selection_policy": (
            "Every case has three semantically connected reference videos that define one coherent target story. "
            "The set is designed for hot-speed-matched comparison between no-cache teacher, TeaCache plus "
            "condition cache, and TeaCache-only baselines."
        ),
        "cases": records,
    }
    (OUT / "dataset_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    (OUT / "README.md").write_text(
        "# Heavy Condition Asset Dataset v6 8 Organic Pexels Hotmatch 32-Aligned\n\n"
        "Eight curated heavy-condition Ref2VA prompt sets. Each case has exactly three Pexels reference videos "
        "and one prompt describing how to merge those videos into a coherent target scene.\n\n"
        "Reference clips are 5 seconds at 24 fps. They preserve source orientation/aspect ratio and are resized to "
        "about 720p with 32-aligned dimensions, without padding to the final 1376x768 output frame.\n",
        encoding="utf-8",
    )
    v5.make_dataset_overview(records)
    print(OUT)
    print(f"cases={len(records)} refs={sum(len(c['references']) for c in records)}")


if __name__ == "__main__":
    main()
